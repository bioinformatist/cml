use crate::{models::stables::STable, TDengine};
use anyhow::Result;
use cml_core::{
    core::register::{Register, TrainData},
    get_placeholders, Handler, Metadata, SharedBatchState,
};
use rand::Rng;
use std::future::Future;
use std::time::{Duration, SystemTime};
use taos::{taos_query::Manager, *};

impl<D: IntoDsn + Clone + std::marker::Sync> Register<Field, Value, Manager<TaosBuilder>>
    for TDengine<D>
{
    fn init_register(
        &self,
        gt_type: Field,
        optional_fields: Option<Vec<Field>>,
        optional_tags: Option<Vec<Field>>,
    ) -> impl Future<Output = Result<()>> + Send {
        let mut fields = vec![
            Field::new("ts", Ty::Timestamp, 8),
            Field::new("is_train", Ty::Bool, 1),
            gt_type,
        ];
        if let Some(f) = optional_fields {
            fields.extend_from_slice(&f);
        }

        let mut tags = vec![Field::new("model_update_time", Ty::Timestamp, 8)];
        if let Some(t) = optional_tags {
            tags.extend_from_slice(&t);
        }

        let stable = STable::new("training_data", fields, tags);
        async {
            let client = self.build().await?;
            stable.init(&client, Some("training_data")).await?;
            Ok(())
        }
    }

    fn register(
        &self,
        metadata: &Metadata<Value>,
        train_data: Vec<TrainData<Value>>,
        batch_state: Option<&SharedBatchState>,
        pool: &Pool<TaosBuilder>,
    ) -> impl Future<Output = Result<()>> + Send {
        async move {
            let taos = pool.get().await?;
            taos.use_database("training_data").await?;
            let mut stmt = Stmt::init(&taos).await?;

            let mut tags = match *metadata.model_update_time() {
                Some(time) => vec![Value::BigInt(time)],
                None => vec![Value::Null(Ty::BigInt)],
            };

            if let Some(t) = &metadata.optional_tags() {
                tags.extend_from_slice(t)
            };
            let tag_placeholder = get_placeholders(&tags);

            let values_to_bind = {
                let batch_state_cvar = batch_state.map(|batch_state| {
                    let (lock, cvar) = &**batch_state;
                    let mut batch_state = lock.lock().unwrap();
                    while batch_state.contains(&metadata.batch) {
                        batch_state = cvar.wait(batch_state).unwrap();
                    }
                    batch_state.insert(metadata.batch.clone());
                    (batch_state, cvar)
                });

                let mut rng = rand::thread_rng();
                let mut current_ts = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .expect("Clock may have gone backwards")
                    .as_nanos() as i64;

                let mut values_to_bind = Vec::<Vec<ColumnView>>::new();
                for data in &train_data {
                    let mut values = vec![
                        ColumnView::from_nanos_timestamp(vec![current_ts]),
                        ColumnView::from_bools(vec![rng.gen::<f32>() >= 0.2]),
                        ColumnView::from(data.gt().clone()),
                    ];
                    if let Some(fields) = data.optional_fields() {
                        values.append(
                            &mut fields
                                .iter()
                                .map(|f| ColumnView::from(f.to_owned()))
                                .collect::<Vec<ColumnView>>(),
                        )
                    }

                    values_to_bind.push(values);
                    current_ts += Duration::from_nanos(1).as_nanos() as i64;
                }

                if let Some((mut batch_state, cvar)) = batch_state_cvar {
                    batch_state.remove(&metadata.batch);
                    cvar.notify_one();
                }

                values_to_bind
            };

            let field_placeholder = get_placeholders(values_to_bind.first().unwrap());

            stmt.prepare(&format!(
                "INSERT INTO ? USING training_data TAGS ({}) VALUES ({})",
                tag_placeholder, field_placeholder
            ))
            .await?;

            stmt.set_tbname_tags(metadata.batch(), &tags).await?;

            for values in values_to_bind {
                stmt.bind(&values).await?;
            }
            stmt.add_batch().await?;
            stmt.execute().await?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{Arc, Condvar, Mutex},
    };

    use super::*;
    use crate::models::databases::{
        options::{CacheModel, ReplicaNum, SingleSTable},
        Database,
    };
    use cml_core::{core::register::TrainData, Metadata};

    #[tokio::test]
    async fn test_register_init() -> Result<()> {
        let cml = TDengine::from_dsn("taos://");
        let taos = cml.build().await?;

        taos::AsyncQueryable::exec(&taos, "DROP DATABASE IF EXISTS training_data").await?;

        let db = Database::builder()
            .name("training_data")
            .duration(1)
            .keep(90)
            .replica(ReplicaNum::NoReplica)
            .cache_model(CacheModel::None)
            .single_stable(SingleSTable::True)
            .build();
        db.init(&cml.build().await?, None).await?;

        cml.init_register(
            Field::new("gt", Ty::Float, 8),
            Some(vec![
                Field::new("fucking_field_1", Ty::VarChar, 8),
                Field::new("fucking_field_2", Ty::TinyInt, 8),
            ]),
            Some(vec![
                Field::new("fucking_tag_1", Ty::VarChar, 8),
                Field::new("fucking_tag_2", Ty::TinyInt, 8),
            ]),
        )
        .await?;

        assert_eq!(
            taos_query::AsyncFetchable::to_records(
                &mut taos::AsyncQueryable::query(&taos, "SHOW training_data.STABLES").await?
            )
            .await?[0][0],
            taos::Value::VarChar("training_data".to_owned())
        );

        taos::AsyncQueryable::exec(&taos, "DROP DATABASE IF EXISTS training_data").await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_register() -> Result<()> {
        let cml = Arc::new(TDengine::from_dsn("taos://"));
        let pool = Arc::new(cml.build_pool());
        let taos = pool.get().await?;

        taos.exec("DROP DATABASE IF EXISTS training_data").await?;
        taos.exec(
            "CREATE DATABASE IF NOT EXISTS training_data 
            PRECISION 'ns'",
        )
        .await?;
        taos.exec(
            "CREATE STABLE IF NOT EXISTS training_data.training_data
            (ts TIMESTAMP, is_train BOOL, gt FLOAT, fucking_field_1 BINARY(255), fucking_field_2 TINYINT)
            TAGS (model_update_time TIMESTAMP, fucking_tag_1 BINARY(255), fucking_tag_2 TINYINT)"
        ).await?;

        let batch_state = Arc::new((Mutex::new(HashSet::new()), Condvar::new()));

        let model_update_time = (SystemTime::now() - Duration::from_secs(86400))
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;

        let metadata = Arc::new(
            Metadata::builder()
                .model_update_time(model_update_time)
                .batch("Fuck_1".to_owned())
                .optional_tags(vec![
                    Value::VarChar("fuck_1".to_string()),
                    Value::TinyInt(8),
                ])
                .build(),
        );

        let data_1 = vec![
            TrainData::builder()
                .gt(Value::Float(51.8))
                .optional_fields(vec![Value::VarChar("fuck".to_string()), Value::TinyInt(18)])
                .build(),
            TrainData::builder()
                .gt(Value::Float(1.8))
                .optional_fields(vec![Value::VarChar("fuck".to_string()), Value::TinyInt(18)])
                .build(),
        ];

        let data_2 = vec![
            TrainData::builder()
                .gt(Value::Float(51.8))
                .optional_fields(vec![Value::VarChar("fuck".to_string()), Value::TinyInt(18)])
                .build(),
            TrainData::builder()
                .gt(Value::Float(1.8))
                .optional_fields(vec![Value::VarChar("fuck".to_string()), Value::TinyInt(18)])
                .build(),
        ];

        let another_cml = cml.clone();
        let another_metadata = metadata.clone();
        let another_batch_state = batch_state.clone();
        let another_pool = pool.clone();

        let task1 = tokio::spawn(async move {
            cml.register(&metadata, data_1, Some(&batch_state), &pool)
                .await
                .expect("Task 1 failed")
        });
        let task2 = tokio::spawn(async move {
            another_cml
                .register(
                    &another_metadata,
                    data_2,
                    Some(&another_batch_state),
                    &another_pool,
                )
                .await
                .expect("Task 2 failed")
        });

        let (_, _) = tokio::join!(task1, task2);

        let records = taos
            .query_one("SELECT COUNT(*) FROM training_data.training_data")
            .await?
            .unwrap_or(0);
        assert_eq!(4, records);

        taos.exec("DROP DATABASE IF EXISTS training_data").await?;
        Ok(())
    }
}

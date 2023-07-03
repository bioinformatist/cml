use crate::{models::stables::STable, TDengine};
use anyhow::Result;
use cml_core::{
    core::register::{Register, SharedBatchState, TrainData},
    handler::Handler,
    metadata::MetaData,
};
use rand::Rng;
use std::time::{Duration, SystemTime};
use taos::{taos_query::Manager, *};

impl<D: IntoDsn + Clone> Register<Field, Value, Manager<TaosBuilder>> for TDengine<D> {
    async fn init_register(
        &self,
        gt_type: Field,
        optional_fields: Option<Vec<Field>>,
        optional_tags: Option<Vec<Field>>,
    ) -> Result<()> {
        let mut fields = vec![
            Field::new("ts", Ty::Timestamp, 8),
            Field::new("is_train", Ty::Bool, 8),
            Field::new("data_path", Ty::NChar, 255),
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

        let client = self.build().await?;
        stable.init(&client, Some("training_data")).await?;

        Ok(())
    }

    async fn register(
        &self,
        metadata: MetaData<Value>,
        train_data: Vec<TrainData<Value>>,
        batch_state: SharedBatchState,
        pool: &Pool<TaosBuilder>,
    ) -> Result<()> {
        let taos = pool.get().await?;
        taos.use_database("training_data").await?;
        let mut stmt = Stmt::init(&taos)?;

        let (tag_placeholder, field_placeholder) = metadata.get_placeholders();

        stmt.prepare(format!(
            "INSERT INTO ? USING training_data TAGS ({}) VALUES ({})",
            tag_placeholder, field_placeholder
        ))?;

        let mut tags = vec![Value::BigInt(*metadata.model_update_time())];
        if let Some(t) = &metadata.optional_tags() {
            tags.extend_from_slice(t)
        };
        stmt.set_tbname_tags(metadata.batch(), &tags)?;

        let batch_state = batch_state.lock().unwrap();
        batch_state.map.insert(metadata.batch().to_owned(), true);

        let mut rng = rand::thread_rng();

        let mut current_ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("Clock may have gone backwards")
            .as_nanos() as i64;

        for data in &train_data {
            let mut values = vec![
                ColumnView::from_nanos_timestamp(vec![current_ts]),
                ColumnView::from_bools(vec![rng.gen::<f32>() > 0.2]),
                ColumnView::from_nchar(vec![data.data_path().as_path().to_str().unwrap()]),
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

            stmt.bind(&values)?;
            current_ts += Duration::from_nanos(1).as_nanos() as i64;
        }

        stmt.add_batch()?;
        stmt.execute()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::databases::{
        options::{CacheModel, ReplicaNum, SingleSTable},
        DatabaseBuilder,
    };
    use cml_core::{
        core::register::{BatchState, TrainDataBuilder},
        handler::Handler,
        metadata::MetaDataBuilder,
    };

    #[tokio::test]
    async fn test_register_init() -> Result<()> {
        let cml = TDengine::from_dsn("taos://");
        let taos = cml.build().await?;

        taos::AsyncQueryable::exec(&taos, "DROP DATABASE IF EXISTS training_data").await?;

        let db = DatabaseBuilder::default()
            .name("training_data")
            .duration(1)
            .keep(90)
            .replica(ReplicaNum::NoReplica)
            .cache_model(CacheModel::None)
            .single_stable(SingleSTable::True)
            .build()?;
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

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_register() -> Result<()> {
        let taos = TaosBuilder::from_dsn("taos://")?.build().await?;
        taos.exec("DROP DATABASE IF EXISTS training_data").await?;
        taos.exec(
            "CREATE DATABASE IF NOT EXISTS training_data 
            PRECISION 'ns'",
        )
        .await?;
        taos.exec(
            "CREATE STABLE IF NOT EXISTS training_data.training_data
            (ts TIMESTAMP, is_train BOOL, data_path NCHAR(255), gt FLOAT, fucking_field_1 BINARY(255), fucking_field_2 TINYINT)
            TAGS (model_update_time TIMESTAMP, fucking_tag_1 BINARY(255), fucking_tag_2 TINYINT)"
        ).await?;

        let cml = TDengine::from_dsn("taos://");
        let pool = cml.build_pool();
        let batch_state = BatchState::create(2);

        let model_update_time = (SystemTime::now() - Duration::from_secs(86400))
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;

        let batch_meta_1 = MetaDataBuilder::default()
            .model_update_time(model_update_time)
            .batch("Fuck_1".to_owned())
            .inherent_field_num(4)
            .inherent_tag_num(1)
            .optional_field_num(2)
            .optional_tags(Some(vec![
                Value::VarChar("fuck_1".to_string()),
                Value::TinyInt(8),
            ]))
            .build()?;

        let batch_meta_2 = MetaDataBuilder::default()
            .model_update_time(model_update_time)
            .batch("Fuck_2".to_owned())
            .inherent_field_num(4)
            .inherent_tag_num(1)
            .optional_field_num(2)
            .optional_tags(Some(vec![
                Value::VarChar("fuck_2".to_string()),
                Value::TinyInt(8),
            ]))
            .build()?;

        let batch_data_1 = vec![
            TrainDataBuilder::default()
                .data_path("/fuck/your/data_1".into())
                .gt(Value::Float(51.8))
                .optional_fields(Some(vec![
                    Value::VarChar("fuck".to_string()),
                    Value::TinyInt(18),
                ]))
                .build()?,
            TrainDataBuilder::default()
                .data_path("/fuck/your/data_1".into())
                .gt(Value::Float(1.8))
                .optional_fields(Some(vec![
                    Value::VarChar("fuck".to_string()),
                    Value::TinyInt(18),
                ]))
                .build()?,
        ];

        let batch_data_2 = vec![
            TrainDataBuilder::default()
                .data_path("/fuck/your/data_2".into())
                .gt(Value::Float(51.8))
                .optional_fields(Some(vec![
                    Value::VarChar("fuck".to_string()),
                    Value::TinyInt(18),
                ]))
                .build()?,
            TrainDataBuilder::default()
                .data_path("/fuck/your/data_2".into())
                .gt(Value::Float(1.8))
                .optional_fields(Some(vec![
                    Value::VarChar("fuck".to_string()),
                    Value::TinyInt(18),
                ]))
                .build()?,
        ];

        tokio::spawn(async move {
            cml.register(batch_meta_1, batch_data_1, batch_state.clone(), &pool)
                .await
                .unwrap();
            cml.register(batch_meta_2, batch_data_2, batch_state.clone(), &pool)
                .await
                .unwrap();
        })
        .await?;

        let mut result = taos
            .query("SELECT COUNT(*) FROM training_data.training_data")
            .await?;
        let records = result.to_records().await?;
        assert_eq!(vec![vec![Value::BigInt(4)]], records);

        taos.exec("DROP DATABASE IF EXISTS training_data").await?;
        Ok(())
    }
}

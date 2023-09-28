use crate::{models::stables::STable, TDengine};
use anyhow::Result;
use cml_core::{
    core::inference::{Inference, NewSample},
    get_placeholders, Handler, Metadata, SharedBatchState,
};
use std::time::{Duration, SystemTime};
use taos::{taos_query::Manager, *};

impl<D: IntoDsn + Clone> Inference<Field, Value, i64, Manager<TaosBuilder>> for TDengine<D> {
    async fn init_inference(
        &self,
        target_type: Field,
        optional_fields: Option<Vec<Field>>,
        optional_tags: Option<Vec<Field>>,
    ) -> Result<()> {
        let mut fields = vec![Field::new("ts", Ty::Timestamp, 8), target_type];
        if let Some(f) = optional_fields {
            fields.extend_from_slice(&f);
        }

        let mut tags = vec![Field::new("model_update_time", Ty::Timestamp, 8)];
        if let Some(t) = optional_tags {
            tags.extend_from_slice(&t);
        }

        let stable = STable::new("inference", fields, tags);

        let client = self.build().await?;
        stable.init(&client, Some("inference")).await?;

        Ok(())
    }

    async fn inference<FN>(
        &self,
        metadata: &Metadata<Value>,
        available_status: &[&str],
        mut data: Vec<NewSample<Value>>,
        batch_state: &SharedBatchState,
        pool: &Pool<TaosBuilder>,
        inference_fn: FN,
    ) -> Result<()>
    where
        FN: FnOnce(&mut Vec<NewSample<Value>>, &str, i64),
    {
        let taos = pool.get().await?;
        let mut stmt = Stmt::init(&taos).await?;
        taos.use_database("task").await?;
        let last_task_time = taos
            .query_one(format!(
                "SELECT LAST(ts) FROM task.`{}` WHERE status IN ({})",
                metadata.batch(),
                available_status
                    .iter()
                    .map(|s| format!("'{}'", s))
                    .collect::<Vec<String>>()
                    .join(", ")
            ))
            .await?
            .unwrap_or_else(|| panic!("There is no task in batch: {}", metadata.batch()));
        inference_fn(&mut data, metadata.batch(), last_task_time);
        taos.use_database("inference").await?;
        let mut tags = match *metadata.model_update_time() {
            Some(time) => vec![Value::BigInt(time)],
            None => vec![Value::Null(Ty::BigInt)],
        };
        if let Some(t) = &metadata.optional_tags() {
            tags.extend_from_slice(t)
        };
        let tag_placeholder = get_placeholders(&tags);
        let null = ColumnView::null(
            1,
            taos.query("SELECT * FROM inference LIMIT 0")
                .await?
                .fields()
                .get(1)
                .unwrap()
                .ty(),
        );
        let values_to_bind = {
            let (lock, cvar) = &**batch_state;
            let mut batch_state = lock.lock().unwrap();
            while batch_state.contains(&metadata.batch) {
                batch_state = cvar.wait(batch_state).unwrap();
            }
            batch_state.insert(metadata.batch.clone());
            let mut current_ts = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("Clock may have gone backwards")
                .as_nanos() as i64;

            let mut values_to_bind = Vec::<Vec<ColumnView>>::new();
            for sample in data {
                let output = match sample.output() {
                    Some(value) => ColumnView::from(value.clone()),
                    None => null.clone(),
                };
                let mut values = vec![ColumnView::from_nanos_timestamp(vec![current_ts]), output];
                if let Some(fields) = sample.optional_fields() {
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

            batch_state.remove(&metadata.batch);
            cvar.notify_one();
            values_to_bind
        };
        let field_placeholder = get_placeholders(values_to_bind.first().unwrap());
        stmt.prepare(&format!(
            "INSERT INTO ? USING inference TAGS ({}) VALUES ({})",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::databases::{
        options::{CacheModel, ReplicaNum, SingleSTable},
        Database,
    };
    use anyhow::Ok;
    use cml_core::{core::inference::NewSample, Metadata};
    use std::{
        collections::HashSet,
        env, fs,
        io::Write,
        sync::{Arc, Condvar, Mutex},
        time::{Duration, SystemTime},
    };
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_inference_init() -> Result<()> {
        let cml = TDengine::from_dsn("taos://");
        let taos = cml.build().await?;

        taos::AsyncQueryable::exec(&taos, "DROP DATABASE IF EXISTS inference").await?;

        let db = Database::builder()
            .name("inference")
            .duration(1)
            .keep(90)
            .replica(ReplicaNum::NoReplica)
            .cache_model(CacheModel::None)
            .single_stable(SingleSTable::True)
            .build();

        db.init(&cml.build().await?, None).await?;

        cml.init_inference(Field::new("output", Ty::Float, 8), None, None)
            .await?;

        assert_eq!(
            taos_query::AsyncFetchable::to_records(
                &mut taos::AsyncQueryable::query(&taos, "SHOW inference.STABLES").await?
            )
            .await?[0][0],
            taos::Value::VarChar("inference".to_owned())
        );

        taos::AsyncQueryable::exec(&taos, "DROP DATABASE IF EXISTS inference").await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_inference() -> Result<()> {
        let cml = Arc::new(TDengine::from_dsn("taos://"));
        let pool = Arc::new(cml.build_pool());
        let taos = pool.get().await?;

        taos.exec("DROP DATABASE IF EXISTS inference").await?;
        taos.exec("DROP DATABASE IF EXISTS task").await?;
        taos.exec(
            "CREATE DATABASE IF NOT EXISTS inference 
            PRECISION 'ns'",
        )
        .await?;
        taos.exec("CREATE DATABASE IF NOT EXISTS task PRECISION 'ns'")
            .await?;
        taos.exec(
            "CREATE STABLE IF NOT EXISTS inference.inference
            (ts TIMESTAMP, output FLOAT, data_path NCHAR(255))
            TAGS (model_update_time TIMESTAMP)",
        )
        .await?;
        taos.exec(
            "CREATE STABLE IF NOT EXISTS task.task
            (ts TIMESTAMP, status BINARY(8))
            TAGS (model_update_time TIMESTAMP)",
        )
        .await?;
        taos.exec(
            "INSERT INTO task.`FUCK`
            USING task.task
            TAGS ('2022-08-08 18:18:18.518')
            VALUES (NOW, 'TRAIN')",
        )
        .await?;
        taos.exec(
            "INSERT INTO task.`FUCK`
            USING task.task
            TAGS ('2022-08-08 18:18:18.518')
            VALUES (NOW-2s, 'SUCCESS')",
        )
        .await?;
        taos.exec(
            "INSERT INTO task.`FUCK8`
            USING task.task
            TAGS ('2022-08-08 18:18:18.518')
            VALUES (NOW, 'SUCCESS')",
        )
        .await?;

        let model_update_time = (SystemTime::now() - Duration::from_secs(86400))
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        let batch_meta_1: Metadata<Value> = Metadata::builder()
            .model_update_time(model_update_time)
            .batch("FUCK".to_owned())
            .build();
        let batch_meta_2 = Arc::new(
            Metadata::builder()
                .model_update_time(model_update_time)
                .batch("FUCK8".to_owned())
                .build(),
        );
        let mut inference_file1 = NamedTempFile::new()?;
        write!(inference_file1, "8.8")?;
        let mut inference_file2 = NamedTempFile::new()?;
        write!(inference_file2, "98.8")?;

        let batch_data_1 = vec![
            NewSample::builder()
                .optional_fields(vec![Value::NChar(
                    inference_file1.path().to_str().unwrap().to_owned(),
                )])
                .build(),
            NewSample::builder()
                .optional_fields(vec![Value::NChar(
                    inference_file2.path().to_str().unwrap().to_owned(),
                )])
                .build(),
        ];
        let batch_data_2 = vec![NewSample::builder()
            .optional_fields(vec![Value::NChar(
                inference_file1.path().to_str().unwrap().to_owned(),
            )])
            .build()];
        let batch_data_3 = vec![NewSample::builder()
            .optional_fields(vec![Value::NChar(
                inference_file2.path().to_str().unwrap().to_owned(),
            )])
            .build()];
        let available_status = ["SUCCESS"];
        let last_batch_time_1: i64 = taos
            .query_one(format!(
                "SELECT LAST(ts) FROM task.`{}` WHERE status IN ({}) ",
                batch_meta_1.batch(),
                available_status
                    .iter()
                    .map(|s| format!("'{}'", s))
                    .collect::<Vec<String>>()
                    .join(", ")
            ))
            .await?
            .unwrap();
        let last_batch_time_2: i64 = taos
            .query_one(format!(
                "SELECT LAST(ts) FROM task.`{}` WHERE status IN ({}) ",
                batch_meta_2.batch(),
                available_status
                    .iter()
                    .map(|s| format!("'{}'", s))
                    .collect::<Vec<String>>()
                    .join(", ")
            ))
            .await?
            .unwrap();
        let working_dir = env::temp_dir().join("inference_dir");
        fs::create_dir_all(&working_dir)?;
        fs::write(
            working_dir
                .join(batch_meta_1.batch().to_string() + &last_batch_time_1.to_string() + ".txt"),
            b"10",
        )?;
        fs::write(
            working_dir
                .join(batch_meta_2.batch().to_string() + &last_batch_time_2.to_string() + ".txt"),
            b"20",
        )?;
        let inference_fn = |vec_data: &mut Vec<NewSample<Value>>, batch: &str, task_time: i64| {
            // let mut result: Vec<NewSample<Value>> = Vec::new();
            let working_dir = env::temp_dir().join("inference_dir/");
            let model_inference = fs::read_to_string(
                working_dir.join(batch.to_string() + &task_time.to_string() + ".txt"),
            )
            .unwrap()
            .parse::<f32>()
            .unwrap();
            vec_data.iter_mut().for_each(|inference_data| {
                let tmp = fs::read_to_string(
                    inference_data
                        .optional_fields()
                        .as_ref()
                        .unwrap()
                        .first()
                        .unwrap()
                        .to_string()
                        .unwrap(),
                )
                .unwrap()
                .parse::<f32>()
                .unwrap()
                    + model_inference;
                inference_data.output = if tmp > 25.0 {
                    Some(Value::Float(tmp))
                } else {
                    Some(Value::Null(Ty::Float))
                };
            });
        };
        let batch_state = Arc::new((Mutex::new(HashSet::new()), Condvar::new()));
        let cml_2 = cml.clone();
        let batch_state_2 = batch_state.clone();
        let pool_2 = pool.clone();
        let cml_3 = cml.clone();
        let batch_meta_2_dup = batch_meta_2.clone();
        let batch_state_3 = batch_state.clone();
        let pool_3 = pool.clone();
        let task1 = tokio::spawn({
            async move {
                cml.inference(
                    &batch_meta_1,
                    &available_status,
                    batch_data_1,
                    &batch_state,
                    &pool,
                    inference_fn,
                )
                .await
                .expect("Task 1 failed")
            }
        });
        let task2 = tokio::spawn({
            async move {
                cml_2
                    .inference(
                        &batch_meta_2,
                        &available_status,
                        batch_data_2,
                        &batch_state_2,
                        &pool_2,
                        inference_fn,
                    )
                    .await
                    .expect("Task 2 failed")
            }
        });

        let task3 = tokio::spawn({
            async move {
                cml_3
                    .inference(
                        &batch_meta_2_dup,
                        &available_status,
                        batch_data_3,
                        &batch_state_3,
                        &pool_3,
                        inference_fn,
                    )
                    .await
                    .expect("Task 3 failed")
            }
        });

        let (_, _, _) = tokio::join!(task1, task2, task3);

        let mut result = taos
            .query("SELECT output FROM inference.inference ORDER BY output ASC")
            .await?;
        let records = result.to_records().await?;
        assert_eq!(
            vec![
                vec![Value::Null(Ty::Float)],
                vec![Value::Float(28.8)],
                vec![Value::Float(108.8)],
                vec![Value::Float(118.8)]
            ],
            records
        );

        fs::remove_dir_all(working_dir)?;
        taos.exec("DROP DATABASE IF EXISTS inference").await?;
        taos.exec("DROP DATABASE IF EXISTS task").await?;
        Ok(())
    }
}

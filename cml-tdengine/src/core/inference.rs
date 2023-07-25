use crate::{models::stables::STable, TDengine};
use anyhow::Result;
use cml_core::{
    core::inference::{Inference, NewSample},
    handler::Handler,
    metadata::MetaData,
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
        let mut fields = vec![
            Field::new("ts", Ty::Timestamp, 8),
            Field::new("data_path", Ty::NChar, 255),
            target_type,
        ];
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
        metadata: MetaData<Value>,
        target_type: Field,
        available_status: &[&str],
        data: &mut Vec<NewSample<Value>>,
        pool: &Pool<TaosBuilder>,
        inference_fn: FN,
    ) -> Result<()>
    where
        FN: FnOnce(&mut Vec<NewSample<Value>>, &str, i64) -> Vec<NewSample<Value>>,
    {
        let taos = pool.get().await?;
        let mut stmt = Stmt::init(&taos)?;
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
        let samples_with_res = inference_fn(data, metadata.batch(), last_task_time);
        taos.use_database("inference").await?;
        let (tag_placeholder, field_placeholder) = metadata.get_placeholders();

        stmt.prepare(format!(
            "INSERT INTO ? USING inference TAGS ({}) VALUES ({})",
            tag_placeholder, field_placeholder
        ))?;

        let mut tags = vec![Value::BigInt(*metadata.model_update_time())];
        if let Some(t) = &metadata.optional_tags() {
            tags.extend_from_slice(t)
        };
        stmt.set_tbname_tags(metadata.batch(), &tags)?;

        let mut current_ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("Clock may have gone backwards")
            .as_nanos() as i64;

        for sample in &samples_with_res {
            let output = match sample.output() {
                Some(value) => ColumnView::from(value.clone()),
                None => ColumnView::null(1, target_type.ty()),
            };

            let mut values = vec![
                ColumnView::from_nanos_timestamp(vec![current_ts]),
                ColumnView::from_nchar(vec![sample.data_path().as_path().to_str().unwrap()]),
                output,
            ];
            if let Some(fields) = sample.optional_fields() {
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
        core::inference::NewSampleBuilder, handler::Handler, metadata::MetaDataBuilder,
    };
    use std::fs;
    use std::time::{Duration, SystemTime};

    #[tokio::test]
    async fn test_inference_init() -> Result<()> {
        let cml = TDengine::from_dsn("taos://");
        let taos = cml.build().await?;

        taos::AsyncQueryable::exec(&taos, "DROP DATABASE IF EXISTS inference").await?;

        let db = DatabaseBuilder::default()
            .name("inference")
            .duration(1)
            .keep(90)
            .replica(ReplicaNum::NoReplica)
            .cache_model(CacheModel::None)
            .single_stable(SingleSTable::True)
            .build()?;

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
        let cml = TDengine::from_dsn("taos://");
        let pool = cml.build_pool();
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
            (ts TIMESTAMP, data_path NCHAR(255), output FLOAT)
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
        let batch_meta_1: MetaData<Value> = MetaDataBuilder::default()
            .model_update_time(model_update_time)
            .batch("FUCK".to_owned())
            .inherent_field_num(3)
            .inherent_tag_num(1)
            .optional_field_num(0)
            .build()?;
        let batch_meta_2: MetaData<Value> = MetaDataBuilder::default()
            .model_update_time(model_update_time)
            .batch("FUCK8".to_owned())
            .inherent_field_num(3)
            .inherent_tag_num(1)
            .optional_field_num(0)
            .build()?;

        fs::create_dir_all("/tmp/inference_dir/")?;
        fs::write("/tmp/inference_dir/inference_data1.txt", b"8.8")?;
        fs::write("/tmp/inference_dir/inference_data2.txt", b"98.8")?;
        let mut batch_data_1 = vec![
            NewSampleBuilder::default()
                .data_path("/tmp/inference_dir/inference_data1.txt".into())
                .build()?,
            NewSampleBuilder::default()
                .data_path("/tmp/inference_dir/inference_data2.txt".into())
                .build()?,
        ];
        let mut batch_data_2 = vec![NewSampleBuilder::default()
            .data_path("/tmp/inference_dir/inference_data1.txt".into())
            .build()?];

        let available_status = vec!["SUCCESS"];
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
        fs::write(
            "/tmp/inference_dir/".to_string()
                + batch_meta_1.batch()
                + &last_batch_time_1.to_string()
                + ".txt",
            b"10",
        )?;
        fs::write(
            "/tmp/inference_dir/".to_string()
                + batch_meta_2.batch()
                + &last_batch_time_2.to_string()
                + ".txt",
            b"20",
        )?;
        let inference_fn = |vec_data: &mut Vec<NewSample<Value>>,
                            batch: &str,
                            task_time: i64|
         -> Vec<NewSample<Value>> {
            let mut result: Vec<NewSample<Value>> = Vec::new();
            let working_dir = "/tmp/inference_dir/".to_string();
            let model_inference =
                fs::read_to_string(working_dir + batch + &task_time.to_string() + ".txt")
                    .unwrap()
                    .parse::<f32>()
                    .unwrap();
            for inference_data in vec_data.iter() {
                // inference
                let output = fs::read_to_string(inference_data.data_path())
                    .unwrap()
                    .parse::<f32>()
                    .unwrap()
                    + model_inference;
                result.push(
                    NewSampleBuilder::default()
                        .data_path(inference_data.data_path().to_path_buf())
                        .output(Some(Value::Float(output)))
                        .build()
                        .unwrap(),
                );
            }
            result
        };

        tokio::spawn({
            async move {
                cml.inference(
                    batch_meta_1,
                    Field::new("output", Ty::Float, 8),
                    &available_status,
                    &mut batch_data_1,
                    &pool,
                    inference_fn,
                )
                .await
                .unwrap();
                cml.inference(
                    batch_meta_2,
                    Field::new("output", Ty::Float, 8),
                    &available_status,
                    &mut batch_data_2,
                    &pool,
                    inference_fn,
                )
                .await
                .unwrap();
            }
        })
        .await?;

        let mut result = taos
            .query("SELECT output FROM inference.inference ORDER BY output ASC")
            .await?;
        let records = result.to_records().await?;
        assert_eq!(
            vec![
                vec![Value::Float(18.8)],
                vec![Value::Float(28.8)],
                vec![Value::Float(108.8)]
            ],
            records
        );

        fs::remove_dir_all("/tmp/inference_dir/")?;
        taos.exec("DROP DATABASE IF EXISTS inference").await?;
        taos.exec("DROP DATABASE IF EXISTS task").await?;
        Ok(())
    }
}

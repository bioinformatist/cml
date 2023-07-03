use anyhow::Result;
use cml_core::{
    core::inference::{Inference, NewSample},
    handler::Handler,
    metadata::MetaData,
};
use std::future::Future;
use std::time::{Duration, SystemTime};
use taos::{taos_query::Manager, *};

use crate::{models::stables::STable, TDengine};

impl<D: IntoDsn + Clone> Inference<Field, Value, Manager<TaosBuilder>> for TDengine<D> {
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

    async fn inference<FN, R>(
        &self,
        metadata: MetaData<Value>,
        target_type: Field,
        data: &mut Vec<NewSample<Value>>,
        pool: &Pool<TaosBuilder>,
        inference_fn: FN,
    ) -> Result<()>
    where
        FN: FnOnce(&mut Vec<NewSample<Value>>, &Pool<TaosBuilder>) -> R,
        R: Future<Output = Result<Vec<NewSample<Value>>>>,
    {
        let samples_with_res = inference_fn(data, pool).await?;

        let taos = pool.get().await?;
        taos.use_database("training_data").await?;
        let mut stmt = Stmt::init(&taos)?;

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
        Ok(())
    }
}

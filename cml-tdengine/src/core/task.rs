use crate::{models::stables::STable, TDengine};
use anyhow::Result;
use chrono::{DateTime, Local};
use rayon::prelude::*;
use serde::Deserialize;

use cml_core::{
    core::task::{Task, TaskConfig},
    handler::Handler,
};
use taos::sync::*;

#[derive(Deserialize)]
struct BatchInfo {
    #[serde(alias = "tbname")]
    batch: String,
    model_update_time: Option<DateTime<Local>>,
}

#[derive(Deserialize)]
struct TaskInfo {
    #[serde(alias = "tbname")]
    batch: String,
    ts: DateTime<Local>,
}

impl<D: IntoDsn + Clone> Task<Field> for TDengine<D> {
    async fn init_task(
        &self,
        optional_fields: Option<Vec<Field>>,
        optional_tags: Option<Vec<Field>>,
    ) -> Result<()> {
        let mut fields = vec![
            Field::new("ts", Ty::Timestamp, 8),
            Field::new("status", Ty::NChar, 16),
        ];
        if let Some(f) = optional_fields {
            fields.extend_from_slice(&f);
        }

        let mut tags = vec![Field::new("model_update_time", Ty::Timestamp, 8)];
        if let Some(t) = optional_tags {
            tags.extend_from_slice(&t);
        }

        let stable = STable::new("task", fields, tags);

        let client = self.build().await?;
        stable.init(&client, Some("task")).await?;

        Ok(())
    }

    fn run<FN>(
        &self,
        task_config: TaskConfig,
        build_from_scratch_fn: FN,
        fining_build_fn: FN,
    ) -> Result<()>
    where
        FN: Fn(&TaskConfig, &str) -> Result<()> + Send + Sync,
    {
        let taos = self.build_sync().unwrap();

        let mut batch_info: Vec<BatchInfo> = taos
            .query("SELECT DISTINCT TBNAME, model_update_time FROM training_data.training_data")?
            .deserialize()
            .try_collect()?;

        let task_info: Vec<TaskInfo> = taos
            .query(format!(
                "SELECT DISTINCT TBNAME, ts FROM task.task WHERE status IN ({})",
                task_config
                    .working_status()
                    .iter()
                    .map(|s| format!("'{}'", s))
                    .collect::<Vec<String>>()
                    .join(", ")
            ))?
            .deserialize()
            .try_collect()?;

        let mut timeout_clause = Vec::<String>::new();
        let mut batch_with_task = Vec::<String>::new();
        for task in task_info {
            if Local::now().signed_duration_since(task.ts) > *task_config.limit_time() {
                timeout_clause.push(format!(
                    "task.`{}` (ts, status) VALUES ({}, '{}') ",
                    task.batch,
                    task.ts.timestamp_nanos(),
                    "DEAD"
                ));
            } else {
                batch_with_task.push(task.batch);
            }
        }

        if !timeout_clause.is_empty() {
            taos.exec("INSERT INTO".to_owned() + &timeout_clause.join(" "))?;
        }
        batch_info.retain(|b| !batch_with_task.contains(&b.batch));

        let mut scratch_in_queue = Vec::<String>::new();
        let mut fining_in_queue = Vec::<String>::new();

        for batch in batch_info {
            match batch.model_update_time {
                Some(model_update_time) => {
                    if let Value::BigInt(count) = taos
                        .query(format!(
                            "SELECT COUNT(*) FROM training_data.`{}` WHERE ts > {}",
                            batch.batch,
                            model_update_time.timestamp_nanos()
                        ))?
                        .to_rows_vec()?[0][0]
                    {
                        if count as usize > *task_config.min_update_count() {
                            scratch_in_queue.push(batch.batch);
                        }
                    }
                }
                None => {
                    if let Value::BigInt(count) = taos
                        .query(format!(
                            "SELECT COUNT(*) FROM training_data.`{}`",
                            batch.batch
                        ))?
                        .to_rows_vec()?[0][0]
                    {
                        if count as usize > *task_config.min_start_count() {
                            fining_in_queue.push(batch.batch);
                        }
                    }
                }
            }
        }

        rayon::join(
            || {
                scratch_in_queue
                    .par_iter()
                    .map(|b| build_from_scratch_fn(&task_config, b).unwrap())
                    .collect::<Vec<()>>()
            },
            || {
                fining_in_queue
                    .par_iter()
                    .map(|b| fining_build_fn(&task_config, b).unwrap())
                    .collect::<Vec<()>>()
            },
        );

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
    use chrono::Duration;
    use cml_core::core::task::TaskConfigBuilder;
    use std::{fs::File, io::prelude::*};

    use burn::{
        data::{
            dataloader::{batcher::Batcher, DataLoaderBuilder},
            dataset::Dataset,
        },
        module::Module,
        nn::{
            self,
            loss::{MSELoss, Reduction},
        },
        optim::{decay::WeightDecayConfig, AdamConfig},
        record::{CompactRecorder, NoStdTrainingRecorder, Recorder},
        tensor::{
            backend::{ADBackend, Backend},
            Data, Tensor,
        },
        train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep},
    };
    use burn_autodiff::ADBackendDecorator;
    use burn_ndarray::{NdArrayBackend, NdArrayDevice};

    #[tokio::test]
    async fn test_task_init() -> Result<()> {
        let cml = TDengine::from_dsn("taos://");
        let taos = cml.build().await?;

        taos::AsyncQueryable::exec(&taos, "DROP DATABASE IF EXISTS task").await?;

        let db = DatabaseBuilder::default()
            .name("task")
            .duration(30)
            .keep(365)
            .replica(ReplicaNum::NoReplica)
            .cache_model(CacheModel::None)
            .single_stable(SingleSTable::True)
            .build()?;
        db.init(&cml.build().await?, None).await?;

        cml.init_task(None, None).await?;

        assert_eq!(
            taos_query::AsyncFetchable::to_records(
                &mut taos::AsyncQueryable::query(&taos, "SHOW task.STABLES").await?
            )
            .await?[0][0],
            taos::Value::VarChar("task".to_owned())
        );

        taos::AsyncQueryable::exec(&taos, "DROP DATABASE IF EXISTS task").await?;
        Ok(())
    }

    #[test]
    fn test_task_running_in_parallel() -> Result<()> {
        let cml = TDengine::from_dsn("taos://");

        let config: TaskConfig = TaskConfigBuilder::default()
            .min_start_count(1)
            .min_update_count(1)
            .work_dir("/tmp/work_dir".into())
            .local_dir(Some("/tmp/local_dir".into()))
            .working_status(vec!["TRAIN".to_string(), "EVAL".to_string()])
            .limit_time(Duration::days(2))
            .build()?;

        for i in 1..=2 {
            let mut file =
                File::create(format!("/tmp/file_{}.txt", i)).expect("Unable to create file");
            write!(file, "8.8").expect("Failed to write to file");
        }

        let taos = cml.build_sync()?;
        taos.exec("DROP DATABASE IF EXISTS training_data")?;
        taos.exec("DROP DATABASE IF EXISTS task")?;

        taos.exec("CREATE DATABASE IF NOT EXISTS training_data PRECISION 'ns'")?;
        taos.exec("CREATE DATABASE IF NOT EXISTS task PRECISION 'ns'")?;

        taos.exec(
            "CREATE STABLE IF NOT EXISTS training_data.training_data
            (ts TIMESTAMP, is_train BOOL, data_path NCHAR(255), gt FLOAT)
            TAGS (model_update_time TIMESTAMP)",
        )?;
        taos.exec(
            "CREATE STABLE IF NOT EXISTS task.task
            (ts TIMESTAMP, status BINARY(8))
            TAGS (model_update_time TIMESTAMP)",
        )?;

        taos.exec(
            "INSERT INTO training_data.`FUCK`
            USING training_data.training_data
            TAGS ('2022-08-08 18:18:18.518')
            VALUES (NOW, 'true', '/tmp/file_1.txt', 1.0),
            (NOW + 1s, 'false', '/tmp/file_2.txt', 2.0)",
        )?;

        let build_fn = |c: &TaskConfig, b: &str| -> Result<()> {
            type B = ADBackendDecorator<NdArrayBackend<f32>>;
            B::seed(220225);

            pub struct DemoBatcher<B: Backend> {
                device: B::Device,
            }

            #[derive(Clone, Debug)]
            pub struct DemoBatch<B: Backend> {
                pub features: Tensor<B, 2>,
                pub targets: Tensor<B, 2>,
            }

            #[derive(Clone, Debug)]
            pub struct DemoItem {
                pub x: [f32; 1],
                pub y: f32,
            }

            impl<B: Backend> DemoBatcher<B> {
                pub fn new(device: B::Device) -> Self {
                    Self { device }
                }
            }

            struct DemoDataset {
                dataset: Vec<DemoItem>,
            }

            impl Dataset<DemoItem> for DemoDataset {
                fn get(&self, index: usize) -> Option<DemoItem> {
                    self.dataset.get(index).cloned()
                }
                fn len(&self) -> usize {
                    self.dataset.len()
                }
            }

            impl DemoDataset {
                fn train(taos: &Taos, batch: &str) -> Self {
                    Self::new(taos, batch, "true").unwrap()
                }

                fn test(taos: &Taos, batch: &str) -> Self {
                    Self::new(taos, batch, "false").unwrap()
                }

                fn new(taos: &Taos, batch: &str, is_train: &str) -> Result<Self> {
                    let records = taos
                        .query(format!(
                            "SELECT data_path, gt FROM training_data.`{}` WHERE is_train = '{}'",
                            batch, is_train
                        ))?
                        .to_rows_vec()?;

                    let mut dataset = Vec::<DemoItem>::new();
                    for record in records {
                        if let [Value::NChar(data_path), Value::Float(gt)] = record.as_slice() {
                            let mut file = File::open(data_path).expect("Failed to open file");
                            let mut contents = String::new();
                            file.read_to_string(&mut contents)
                                .expect("Failed to read from file");

                            dataset.push(DemoItem {
                                x: [contents.trim().parse().expect("Failed to parse number")],
                                y: *gt,
                            })
                        }
                    }

                    Ok(Self { dataset })
                }
            }

            impl<B: Backend> Batcher<DemoItem, DemoBatch<B>> for DemoBatcher<B> {
                fn batch(&self, items: Vec<DemoItem>) -> DemoBatch<B> {
                    let features = items
                        .iter()
                        .map(|item| Data::<f32, 1>::from(item.x))
                        .map(|data| Tensor::<B, 1>::from_data(data.convert()))
                        .map(|tensor| tensor.reshape([1, 1]))
                        .collect();

                    let targets = items
                        .iter()
                        .map(|item| Data::<f32, 1>::from([item.y]))
                        .map(|data| Tensor::<B, 1>::from_data(data.convert()))
                        .map(|tensor| tensor.reshape([1, 1]))
                        .collect();

                    let features = Tensor::cat(features, 0).to_device(&self.device);
                    let targets = Tensor::cat(targets, 0).to_device(&self.device);

                    DemoBatch { features, targets }
                }
            }

            let device = NdArrayDevice::Cpu;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                fc1: nn::Linear<B>,
                fc2: nn::Linear<B>,
                activation: nn::GELU,
            }

            impl<B: Backend> Model<B> {
                pub fn new() -> Self {
                    let fc1 = nn::LinearConfig::new(1, 4).with_bias(false).init();
                    let fc2 = nn::LinearConfig::new(4, 1).with_bias(false).init();

                    Self {
                        fc1,
                        fc2,
                        activation: nn::GELU::new(),
                    }
                }

                pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
                    let [batch_size, feature_num] = input.dims();

                    let x = input.reshape([batch_size, feature_num]).detach();

                    let [batch_size, feature_num] = x.dims();
                    let x = x.reshape([batch_size, feature_num]);

                    let x = self.fc1.forward(x);
                    let x = self.activation.forward(x);

                    self.fc2.forward(x)
                }

                pub fn forward_regression(&self, item: DemoBatch<B>) -> RegressionOutput<B> {
                    let targets = item.targets;
                    let output = self.forward(item.features);
                    let loss = MSELoss::new();
                    let loss = loss.forward(output.clone(), targets.clone(), Reduction::Auto);

                    RegressionOutput {
                        loss,
                        output,
                        targets,
                    }
                }
            }

            impl<B: ADBackend> TrainStep<DemoBatch<B>, RegressionOutput<B>> for Model<B> {
                fn step(&self, item: DemoBatch<B>) -> TrainOutput<RegressionOutput<B>> {
                    let item = self.forward_regression(item);

                    TrainOutput::new(self, item.loss.backward(), item)
                }
            }

            impl<B: Backend> ValidStep<DemoBatch<B>, RegressionOutput<B>> for Model<B> {
                fn step(&self, item: DemoBatch<B>) -> RegressionOutput<B> {
                    self.forward_regression(item)
                }
            }

            let batcher_train = DemoBatcher::<B>::new(device);
            let batcher_valid = DemoBatcher::<<burn_autodiff::ADBackendDecorator<burn_ndarray::NdArrayBackend<f32>> as burn::tensor::backend::ADBackend>::InnerBackend>::new(device);

            let dataloader_train = DataLoaderBuilder::new(batcher_train)
                .batch_size(1)
                .shuffle(220225)
                .num_workers(1)
                .build(DemoDataset::train(&taos, b));
            let dataloader_test = DataLoaderBuilder::new(batcher_valid)
                .batch_size(1)
                .shuffle(220225)
                .num_workers(1)
                .build(DemoDataset::test(&taos, b));

            let config_optimizer =
                AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));

            let working_dir = c.work_dir().to_str().unwrap();

            let learner = LearnerBuilder::new(working_dir)
                .with_file_checkpointer(1, CompactRecorder::new())
                .devices(vec![device])
                .num_epochs(1)
                .build(Model::<B>::new(), config_optimizer.init(), 1e-4);

            let model_trained = learner.fit(dataloader_train, dataloader_test);

            NoStdTrainingRecorder::new()
                .record(
                    model_trained.into_record(),
                    format!("{working_dir}/model").into(),
                )
                .expect("Failed to save trained model");

            Ok(())
        };

        cml.run(config, build_fn, build_fn)?;

        taos.exec("DROP DATABASE IF EXISTS training_data")?;
        taos.exec("DROP DATABASE IF EXISTS task")?;

        Ok(())
    }
}

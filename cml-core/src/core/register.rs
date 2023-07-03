use crate::metadata::MetaData;
use anyhow::Result;
use dashmap::DashMap;
use deadpool::managed::{Manager, Pool};
use derive_getters::Getters;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Builder, Getters)]
pub struct TrainData<F> {
    data_path: PathBuf,
    gt: F,
    #[builder(default = "None")]
    optional_fields: Option<Vec<F>>,
}

pub struct BatchState {
    pub map: DashMap<String, bool>,
}

pub type SharedBatchState = Arc<Mutex<BatchState>>;

impl BatchState {
    pub fn create(num_shards: usize) -> SharedBatchState {
        Arc::new(Mutex::new(BatchState {
            map: DashMap::with_shard_amount(num_shards),
        }))
    }
}

pub trait Register<M, F, C: Manager> {
    async fn init_register(
        &self,
        gt_type: M,
        optional_fields: Option<Vec<M>>,
        optional_tags: Option<Vec<M>>,
    ) -> Result<()>;

    async fn register(
        &self,
        metadata: MetaData<F>,
        train_data: Vec<TrainData<F>>,
        batch_state: Arc<Mutex<BatchState>>,
        pool: &Pool<C>,
    ) -> Result<()>;
}

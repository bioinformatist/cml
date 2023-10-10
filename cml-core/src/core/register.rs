use crate::{metadata::Metadata, SharedBatchState};
use anyhow::Result;
use deadpool::managed::{Manager, Pool};
use derive_getters::Getters;
use std::future::Future;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder, Getters, Clone)]
pub struct TrainData<F> {
    gt: F,
    #[builder(default, setter(strip_option))]
    optional_fields: Option<Vec<F>>,
}

pub trait Register<M, F, C: Manager> {
    fn init_register(
        &self,
        gt_type: M,
        tag_name: &str,
        optional_fields: Option<Vec<M>>,
    ) -> impl Future<Output = Result<()>> + Send;

    fn register(
        &self,
        metadata: &Metadata<F>,
        train_data: Vec<TrainData<F>>,
        batch_state: Option<&SharedBatchState>,
        pool: &Pool<C>,
    ) -> impl Future<Output = Result<()>> + Send;
}

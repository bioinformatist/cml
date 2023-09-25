use crate::{metadata::Metadata, SharedBatchState};
use anyhow::Result;
use deadpool::managed::{Manager, Pool};
use derive_getters::Getters;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder, Getters)]
pub struct TrainData<F> {
    gt: F,
    #[builder(default, setter(strip_option))]
    optional_fields: Option<Vec<F>>,
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
        metadata: &Metadata<F>,
        train_data: Vec<TrainData<F>>,
        batch_state: &SharedBatchState,
        pool: &Pool<C>,
    ) -> Result<()>;
}

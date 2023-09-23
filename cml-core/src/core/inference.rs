use crate::{metadata::Metadata, SharedBatchState};
use anyhow::Result;
use deadpool::managed::{Manager, Pool};
use derive_getters::Getters;
use std::path::PathBuf;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder, Getters)]
pub struct NewSample<F> {
    data_path: PathBuf,
    #[builder(default, setter(strip_option))]
    output: Option<F>,
    #[builder(default, setter(strip_option))]
    optional_fields: Option<Vec<F>>,
    #[builder(default, setter(strip_option))]
    optional_tags: Option<Vec<F>>,
}

pub trait Inference<M, F, T, C: Manager> {
    async fn init_inference(
        &self,
        target_type: M,
        optional_fields: Option<Vec<M>>,
        optional_tags: Option<Vec<M>>,
    ) -> Result<()>;

    async fn inference<FN>(
        &self,
        metadata: Metadata<F>,
        available_status: &[&str],
        data: &mut Vec<NewSample<F>>,
        batch_state: &SharedBatchState,
        pool: &Pool<C>,
        inference_fn: FN,
    ) -> Result<()>
    where
        FN: FnOnce(&mut Vec<NewSample<F>>, &str, T) -> Vec<NewSample<F>>;
}

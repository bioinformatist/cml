use crate::metadata::MetaData;
use anyhow::Result;
use deadpool::managed::{Manager, Pool};
use derive_getters::Getters;
use std::path::PathBuf;

#[derive(Builder, Getters)]
pub struct NewSample<F> {
    data_path: PathBuf,
    #[builder(default = "None")]
    output: Option<F>,
    #[builder(default = "None")]
    optional_fields: Option<Vec<F>>,
    #[builder(default = "None")]
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
        metadata: MetaData<F>,
        target_type: M,
        data: &mut Vec<NewSample<F>>,
        pool: &Pool<C>,
        inference_fn: FN,
    ) -> Result<()>
    where
        FN: FnOnce(&mut Vec<NewSample<F>>, &str, T) -> Vec<NewSample<F>>;
}

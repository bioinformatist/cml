use anyhow::Result;
use std::future::Future;
pub trait Task<M> {
    fn init_task(
        &self,
        optional_fields: Option<Vec<M>>,
        optional_tags: Option<Vec<M>>,
    ) -> impl Future<Output = Result<()>> + Send;

    fn run<FN>(&self, build_from_scratch_fn: FN, fining_build_fn: FN) -> Result<()>
    where
        FN: Fn(&str) -> Result<()> + Send + Sync;
}

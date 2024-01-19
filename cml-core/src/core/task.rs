use anyhow::Result;
use std::future::Future;
pub trait Task<M> {
    fn init_task(
        &self,
        optional_fields: Option<Vec<M>>,
        optional_tags: Option<Vec<M>>,
    ) -> impl Future<Output = Result<()>> + Send;

    fn prepare(&self) -> Result<(Vec<String>, Vec<String>)>;
}

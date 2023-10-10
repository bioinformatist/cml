use anyhow::Result;
use chrono::Duration;
use derive_getters::Getters;
use std::future::Future;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder, Getters, Clone)]
pub struct TaskConfig<'a> {
    min_start_count: usize,
    min_update_count: usize,
    working_status: &'a [String],
    limit_time: Duration,
}

pub trait Task<M> {
    fn init_task(
        &self,
        tag_name: &str,
        optional_fields: Option<Vec<M>>,
    ) -> impl Future<Output = Result<()>> + Send;

    fn run<FN>(
        &self,
        task_config: TaskConfig,
        build_from_scratch_fn: FN,
        fining_build_fn: FN,
    ) -> Result<()>
    where
        FN: Fn(&str) -> Result<()> + Send + Sync;
}

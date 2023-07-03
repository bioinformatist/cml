use anyhow::Result;
use chrono::Duration;
use derive_getters::Getters;
use std::path::PathBuf;

#[derive(Builder, Getters, Clone)]
pub struct TaskConfig {
    min_start_count: usize,
    min_update_count: usize,
    work_dir: PathBuf,
    local_dir: Option<PathBuf>,
    working_status: Vec<String>,
    limit_time: Duration,
}

pub trait Task<M> {
    async fn init_task(
        &self,
        optional_fields: Option<Vec<M>>,
        optional_tags: Option<Vec<M>>,
    ) -> Result<()>;

    fn run<FN>(
        &self,
        task_config: TaskConfig,
        build_from_scratch_fn: FN,
        fining_build_fn: FN,
    ) -> Result<()>
    where
        FN: Fn(&TaskConfig, &str) -> Result<()> + Send + Sync;
}

//! cml_tdengine
//!
//!
use anyhow::Result;
use chrono::Duration;
use derive_getters::Getters;
use taos::taos_query::Manager;
use taos::{sync::*, Pool};

mod core;
mod models;
#[derive(TypedBuilder, Clone, Getters)]
pub struct TDengine<D> {
    dsn: D,
    min_start_count: usize,
    min_update_count: usize,
    working_status: Vec<String>,
    available_status: Vec<String>,
    limit_time: Duration,
}

impl<D: IntoDsn + Clone> TDengine<D> {
    #[allow(dead_code)]
    pub fn new(
        dsn: D,
        min_start_count: usize,
        min_update_count: usize,
        working_status: Vec<String>,
        available_status: Vec<String>,
        limit_time: Duration,
    ) -> Self {
        TDengine {
            dsn,
            min_start_count,
            min_update_count,
            working_status,
            available_status,
            limit_time,
        }
    }

    #[allow(dead_code)]
    pub fn build_pool(&self) -> Pool<TaosBuilder> {
        Pool::builder(Manager::from_dsn(self.dsn.clone()).unwrap().0)
            .max_size(88)
            .build()
            .unwrap()
    }

    async fn build(&self) -> Result<Taos> {
        Ok(
            taos::taos_query::AsyncTBuilder::build(&TaosBuilder::from_dsn(self.dsn.clone())?)
                .await?,
        )
    }

    pub fn build_sync(&self) -> Result<Taos> {
        Ok(TaosBuilder::from_dsn(self.dsn.clone())?.build()?)
    }
}

pub use models::databases::{options::*, Database};
use typed_builder::TypedBuilder;

//! cml_tdengine
//!
//!

#![allow(incomplete_features)]
#![feature(async_fn_in_trait)]

use anyhow::Result;
use taos::taos_query::Manager;
use taos::{sync::*, Pool};

mod core;
mod models;

#[derive(Clone)]
struct TDengine<D> {
    dsn: D,
}

impl<D: IntoDsn + Clone> TDengine<D> {
    #[allow(dead_code)]
    fn from_dsn(dsn: D) -> Self {
        TDengine { dsn }
    }

    #[allow(dead_code)]
    fn build_pool(&self) -> Pool<TaosBuilder> {
        Pool::builder(Manager::from_dsn(self.dsn.clone()).unwrap().0)
            .max_size(88)
            .build()
            .unwrap()
    }

    async fn build(&self) -> Result<Taos> {
        Ok(taos_query::AsyncTBuilder::build(&TaosBuilder::from_dsn(self.dsn.clone())?).await?)
    }

    fn build_sync(&self) -> Result<Taos> {
        Ok(TaosBuilder::from_dsn(self.dsn.clone())?.build()?)
    }
}

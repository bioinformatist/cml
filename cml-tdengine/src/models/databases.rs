pub(crate) mod options;

use anyhow::Result;
use cml_core::Handler;
use options::{CacheModel, ReplicaNum, SingleSTable};
use std::future::Future;
use taos::*;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder)]
pub struct Database<'a> {
    name: &'a str,
    duration: i16,
    keep: i16,
    replica: ReplicaNum,
    cache_model: CacheModel,
    single_stable: SingleSTable,
}

impl Handler for Database<'_> {
    type Database = Taos;
    fn init(
        self,
        client: &Self::Database,
        _db: Option<&str>,
    ) -> impl Future<Output = Result<()>> + Send {
        async move {
            client
                .exec(format!(
                    "CREATE DATABASE IF NOT EXISTS `{}` \
            KEEP            {}
            DURATION        {}
            PRECISION       'ns'
            CACHEMODEL      '{}'
            REPLICA         {}
            SINGLE_STABLE   {}",
                    self.name,
                    self.keep,
                    self.duration,
                    self.cache_model.as_str(),
                    self.replica as u8,
                    self.single_stable as u8
                ))
                .await?;
            Ok(())
        }
    }
}

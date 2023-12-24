use anyhow::Result;
use std::future::Future;

pub trait Handler {
    type Database;
    fn init(
        self,
        client: &Self::Database,
        db: Option<&str>,
    ) -> impl Future<Output = Result<()>> + Send;
}

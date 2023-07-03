use anyhow::Result;

pub trait Handler {
    type Database;
    async fn init(self, client: &Self::Database, db: Option<&str>) -> Result<()>;
}

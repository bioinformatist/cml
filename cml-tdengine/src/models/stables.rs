use anyhow::Result;
use cml_core::Handler;
use std::future::Future;
use taos::*;
pub struct STable<'a> {
    name: &'a str,
    fields: Vec<Field>,
    tags: Vec<Field>,
}

impl<'a> STable<'a> {
    pub fn new(name: &'a str, fields: Vec<Field>, tags: Vec<Field>) -> Self {
        Self { name, fields, tags }
    }
}

fn field_to_stmt(fields: &[Field]) -> String {
    fields
        .iter()
        .map(|f| f.sql_repr().to_string())
        .collect::<Vec<String>>()
        .join(",")
}

impl Handler for STable<'_> {
    type Database = Taos;
    fn init(
        self,
        client: &Self::Database,
        db: Option<&str>,
    ) -> impl Future<Output = Result<()>> + Send {
        async move {
            match db {
                Some(db) => client.use_database(db).await?,
                None => panic!(),
            };
            client
                .exec(format!(
                    "CREATE STABLE IF NOT EXISTS `{}` ({}) TAGS ({})",
                    self.name,
                    field_to_stmt(&self.fields),
                    field_to_stmt(&self.tags)
                ))
                .await?;
            Ok(())
        }
    }
}

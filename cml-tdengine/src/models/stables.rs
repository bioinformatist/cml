use anyhow::Result;
use cml_core::handler::Handler;
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
        .map(|f| format!("{}", f.sql_repr()))
        .collect::<Vec<String>>()
        .join(",")
}

impl Handler for STable<'_> {
    type Database = Taos;
    async fn init(self, client: &Self::Database, db: Option<&str>) -> Result<()> {
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

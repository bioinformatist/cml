use derive_getters::Getters;

#[derive(Builder, Getters)]
pub struct MetaData<F> {
    model_update_time: i64,
    batch: String,
    inherent_field_num: usize,
    inherent_tag_num: usize,
    optional_field_num: usize,
    #[builder(default = "None")]
    optional_tags: Option<Vec<F>>,
}

impl<F> MetaData<F> {
    pub fn get_placeholders(&self) -> (String, String) {
        (
            vec![
                "?";
                self.optional_tags.as_ref().map_or(0, |v| v.len()) + self.inherent_tag_num
            ]
            .join(", "),
            vec!["?"; self.optional_field_num + self.inherent_field_num].join(", "),
        )
    }
}

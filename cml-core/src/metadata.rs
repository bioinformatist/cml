use derive_getters::Getters;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder, Getters, Clone)]
pub struct Metadata<F> {
    model_update_time: i64,
    pub batch: String,
    inherent_field_num: usize,
    inherent_tag_num: usize,
    optional_field_num: usize,
    #[builder(default, setter(strip_option))]
    optional_tags: Option<Vec<F>>,
}

impl<F> Metadata<F> {
    pub fn get_placeholders(&self) -> (String, String) {
        (
            vec!["?"; self.optional_tags.as_ref().map_or(0, |v| v.len()) + self.inherent_tag_num]
                .join(", "),
            vec!["?"; self.optional_field_num + self.inherent_field_num].join(", "),
        )
    }
}

use derive_getters::Getters;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder, Getters, Clone)]
pub struct Metadata<F> {
    #[builder(default, setter(strip_option))]
    model_update_time: Option<i64>,
    pub batch: String,
    #[builder(default, setter(strip_option))]
    optional_tags: Option<Vec<F>>,
}

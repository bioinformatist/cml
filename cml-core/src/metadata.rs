use derive_getters::Getters;
use typed_builder::TypedBuilder;

#[derive(TypedBuilder, Getters, Clone)]
pub struct Metadata<F> {
    pub batch: String,
    #[builder(default, setter(strip_option))]
    optional_tags: Option<Vec<F>>,
}

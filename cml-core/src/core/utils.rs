pub fn get_placeholders<T>(fields: &[T]) -> String {
    vec!["?"; fields.len()].join(", ")
}

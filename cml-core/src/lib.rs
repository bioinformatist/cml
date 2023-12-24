use std::{
    collections::HashSet,
    sync::{Arc, Condvar, Mutex},
};

pub mod core;
mod handler;
mod metadata;

pub type SharedBatchState = Arc<(Mutex<HashSet<String>>, Condvar)>;

pub use core::{register::Register, utils::get_placeholders};
pub use handler::Handler;
pub use metadata::Metadata;

#![allow(incomplete_features)]
#![feature(async_fn_in_trait)]

use std::{
    collections::HashSet,
    sync::{Arc, Condvar, Mutex},
};

pub mod core;
mod handler;
mod metadata;

pub type SharedBatchState = Arc<(Mutex<HashSet<String>>, Condvar)>;

pub use core::register::Register;
pub use handler::Handler;
pub use metadata::Metadata;

#![allow(incomplete_features)]
#![feature(async_fn_in_trait)]

use std::{
    collections::HashSet,
    sync::{Arc, Condvar, Mutex},
};

pub mod core;
pub mod handler;
pub mod metadata;

pub type SharedBatchState = Arc<(Mutex<HashSet<String>>, Condvar)>;

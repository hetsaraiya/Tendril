//! Tendril's execution engine.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

pub mod config;
pub mod cpu_kernels;
pub mod device;
pub mod generate;
pub mod linear;
pub mod model;
pub mod pool;
pub mod sampler;
pub mod testing;
pub mod tokenizer;
pub mod weights;

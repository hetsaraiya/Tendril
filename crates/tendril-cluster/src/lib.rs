//! Tendril cluster: machines joined by encrypted links, running one model as
//! a pipeline and serving it through an OpenAI-compatible API.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::large_enum_variant
)]

pub mod agent;
pub mod calibration;
pub mod coordinator;
pub mod http;
pub mod prefix;
pub mod proto;
pub mod shard;
pub mod token;
pub mod wire;
pub mod worker;

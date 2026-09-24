//! Tendril core: everything needed to decide how a model should run on a
//! pool of machines, without loading any weights.

pub mod advice;
pub mod cluster;
pub mod hardware;
pub mod model;
pub mod planner;
pub mod pool;
pub mod presets;
pub mod units;

pub use cluster::{Cluster, Link};
pub use hardware::{Backend, NodeProfile};
pub use model::{ModelSpec, Quant};
pub use planner::{plan, Goal, Plan, PlanOptions, PlanResult, Workload};
pub use units::Bytes;

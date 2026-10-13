//! Tendril cluster: machines joined by encrypted links, running inference
//! engines (llama.cpp, vLLM, Ollama, …) and serving them through one
//! OpenAI-compatible API.

#![allow(
    clippy::too_many_arguments,
    clippy::large_enum_variant,
    clippy::result_large_err
)]

pub mod agent;
pub mod discovery;
pub mod fetch;
pub mod http;
pub mod machine;
pub mod pool;
pub mod proto;
pub mod token;
pub mod tunnel;
pub mod wire;

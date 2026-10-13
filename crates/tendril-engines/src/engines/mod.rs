//! The engines, one module each.

pub mod exo;
pub mod llamacpp;
pub mod media;
pub mod mlx;
pub mod ollama;
pub mod torch;

use crate::Engine;
use std::sync::Arc;

pub fn all() -> Vec<Arc<dyn Engine>> {
    vec![
        Arc::new(llamacpp::LlamaCpp),
        Arc::new(torch::Vllm),
        Arc::new(torch::Sglang),
        Arc::new(ollama::Ollama),
        Arc::new(mlx::Mlx),
        Arc::new(exo::Exo),
        Arc::new(media::ComfyUi),
        Arc::new(media::SdCpp),
        Arc::new(media::WhisperCpp),
    ]
}

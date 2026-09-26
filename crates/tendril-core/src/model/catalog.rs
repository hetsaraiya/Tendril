//! Built-in catalog of popular open models, so `tendril plan llama-3.1-70b`
//! works offline. Numbers mirror the published `config.json` files; use a
//! HuggingFace id or local path for exact tensor-level accounting.

use super::config::spec_from_config;
use super::ModelSpec;
use serde_json::json;

pub struct CatalogEntry {
    pub alias: &'static str,
    pub repo: &'static str,
    pub family: &'static str,
    config: fn() -> serde_json::Value,
}

impl std::fmt::Debug for CatalogEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CatalogEntry({})", self.alias)
    }
}

impl CatalogEntry {
    pub fn spec(&self) -> ModelSpec {
        spec_from_config(self.repo, "built-in catalog", &(self.config)())
            .expect("catalog entry is valid")
    }
}

macro_rules! dense {
    ($ty:expr, $l:expr, $h:expr, $i:expr, $nh:expr, $kv:expr, $hd:expr, $vocab:expr, $tie:expr, $ctx:expr) => {
        || json!({"model_type": $ty, "num_hidden_layers": $l, "hidden_size": $h, "intermediate_size": $i,
                  "num_attention_heads": $nh, "num_key_value_heads": $kv, "head_dim": $hd,
                  "vocab_size": $vocab, "tie_word_embeddings": $tie, "max_position_embeddings": $ctx,
                  "torch_dtype": "bfloat16"})
    };
    ($ty:expr, $l:expr, $h:expr, $i:expr, $nh:expr, $kv:expr, $hd:expr, $vocab:expr, $tie:expr, $ctx:expr, window $w:expr) => {
        || json!({"model_type": $ty, "num_hidden_layers": $l, "hidden_size": $h, "intermediate_size": $i,
                  "num_attention_heads": $nh, "num_key_value_heads": $kv, "head_dim": $hd,
                  "vocab_size": $vocab, "tie_word_embeddings": $tie, "max_position_embeddings": $ctx,
                  "sliding_window": $w, "torch_dtype": "bfloat16"})
    };
}

macro_rules! e {
    ($alias:expr, $repo:expr, $fam:expr, $cfg:expr) => {
        CatalogEntry {
            alias: $alias,
            repo: $repo,
            family: $fam,
            config: $cfg,
        }
    };
}

pub static CATALOG: &[CatalogEntry] = &[
    e!(
        "smollm2-135m",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "SmolLM2",
        dense!("llama", 30, 576, 1536, 9, 3, 64, 49152, true, 8192)
    ),
    e!(
        "smollm2-360m",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "SmolLM2",
        dense!("llama", 32, 960, 2560, 15, 5, 64, 49152, true, 8192)
    ),
    e!(
        "smollm2-1.7b",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "SmolLM2",
        dense!("llama", 24, 2048, 8192, 32, 32, 64, 49152, true, 8192)
    ),
    e!(
        "tinyllama-1.1b",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "TinyLlama",
        dense!("llama", 22, 2048, 5632, 32, 4, 64, 32000, false, 2048)
    ),
    e!(
        "llama-3.2-1b",
        "meta-llama/Llama-3.2-1B-Instruct",
        "Llama 3",
        dense!("llama", 16, 2048, 8192, 32, 8, 64, 128256, true, 131072)
    ),
    e!(
        "llama-3.2-3b",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Llama 3",
        dense!("llama", 28, 3072, 8192, 24, 8, 128, 128256, true, 131072)
    ),
    e!(
        "llama-3.1-8b",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Llama 3",
        dense!("llama", 32, 4096, 14336, 32, 8, 128, 128256, false, 131072)
    ),
    e!(
        "llama-3.3-70b",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Llama 3",
        dense!("llama", 80, 8192, 28672, 64, 8, 128, 128256, false, 131072)
    ),
    e!(
        "llama-3.1-70b",
        "meta-llama/Llama-3.1-70B-Instruct",
        "Llama 3",
        dense!("llama", 80, 8192, 28672, 64, 8, 128, 128256, false, 131072)
    ),
    e!(
        "llama-3.1-405b",
        "meta-llama/Llama-3.1-405B-Instruct",
        "Llama 3",
        dense!("llama", 126, 16384, 53248, 128, 8, 128, 128256, false, 131072)
    ),
    e!(
        "mistral-7b",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral",
        dense!("mistral", 32, 4096, 14336, 32, 8, 128, 32768, false, 32768)
    ),
    e!(
        "mistral-nemo-12b",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "Mistral",
        dense!("mistral", 40, 5120, 14336, 32, 8, 128, 131072, false, 131072)
    ),
    e!(
        "mistral-small-24b",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "Mistral",
        dense!("mistral", 40, 5120, 32768, 32, 8, 128, 131072, false, 32768)
    ),
    e!(
        "qwen2.5-0.5b",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 24, 896, 4864, 14, 2, 64, 151936, true, 32768)
    ),
    e!(
        "qwen2.5-1.5b",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 28, 1536, 8960, 12, 2, 128, 151936, true, 32768)
    ),
    e!(
        "qwen2.5-3b",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 36, 2048, 11008, 16, 2, 128, 151936, true, 32768)
    ),
    e!(
        "qwen2.5-7b",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 28, 3584, 18944, 28, 4, 128, 152064, false, 32768)
    ),
    e!(
        "qwen2.5-14b",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 48, 5120, 13824, 40, 8, 128, 152064, false, 32768)
    ),
    e!(
        "qwen2.5-32b",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 64, 5120, 27648, 40, 8, 128, 152064, false, 32768)
    ),
    e!(
        "qwen2.5-72b",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 80, 8192, 29568, 64, 8, 128, 152064, false, 32768)
    ),
    e!(
        "qwen2.5-coder-7b",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 28, 3584, 18944, 28, 4, 128, 152064, false, 32768)
    ),
    e!(
        "qwen2.5-coder-32b",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen 2.5",
        dense!("qwen2", 64, 5120, 27648, 40, 8, 128, 152064, false, 32768)
    ),
    e!(
        "qwen3-0.6b",
        "Qwen/Qwen3-0.6B",
        "Qwen 3",
        dense!("qwen3", 28, 1024, 3072, 16, 8, 128, 151936, true, 40960)
    ),
    e!(
        "qwen3-1.7b",
        "Qwen/Qwen3-1.7B",
        "Qwen 3",
        dense!("qwen3", 28, 2048, 6144, 16, 8, 128, 151936, true, 40960)
    ),
    e!(
        "qwen3-4b",
        "Qwen/Qwen3-4B",
        "Qwen 3",
        dense!("qwen3", 36, 2560, 9728, 32, 8, 128, 151936, true, 40960)
    ),
    e!(
        "qwen3-8b",
        "Qwen/Qwen3-8B",
        "Qwen 3",
        dense!("qwen3", 36, 4096, 12288, 32, 8, 128, 151936, false, 40960)
    ),
    e!(
        "qwen3-14b",
        "Qwen/Qwen3-14B",
        "Qwen 3",
        dense!("qwen3", 40, 5120, 17408, 40, 8, 128, 151936, false, 40960)
    ),
    e!(
        "qwen3-32b",
        "Qwen/Qwen3-32B",
        "Qwen 3",
        dense!("qwen3", 64, 5120, 25600, 64, 8, 128, 151936, false, 40960)
    ),
    e!(
        "deepseek-r1-distill-qwen-7b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek R1",
        dense!("qwen2", 28, 3584, 18944, 28, 4, 128, 152064, false, 131072)
    ),
    e!(
        "deepseek-r1-distill-qwen-14b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "DeepSeek R1",
        dense!("qwen2", 48, 5120, 13824, 40, 8, 128, 152064, false, 131072)
    ),
    e!(
        "deepseek-r1-distill-qwen-32b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "DeepSeek R1",
        dense!("qwen2", 64, 5120, 27648, 40, 8, 128, 152064, false, 131072)
    ),
    e!(
        "deepseek-r1-distill-llama-70b",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "DeepSeek R1",
        dense!("llama", 80, 8192, 28672, 64, 8, 128, 128256, false, 131072)
    ),
    e!(
        "gemma-2-2b",
        "google/gemma-2-2b-it",
        "Gemma 2",
        dense!("gemma2", 26, 2304, 9216, 8, 4, 256, 256000, true, 8192, window 4096)
    ),
    e!(
        "gemma-2-9b",
        "google/gemma-2-9b-it",
        "Gemma 2",
        dense!("gemma2", 42, 3584, 14336, 16, 8, 256, 256000, true, 8192, window 4096)
    ),
    e!(
        "gemma-2-27b",
        "google/gemma-2-27b-it",
        "Gemma 2",
        dense!("gemma2", 46, 4608, 36864, 32, 16, 128, 256000, true, 8192, window 4096)
    ),
    e!(
        "gemma-3-1b",
        "google/gemma-3-1b-it",
        "Gemma 3",
        dense!("gemma3_text", 26, 1152, 6912, 4, 1, 256, 262144, true, 32768, window 512)
    ),
    e!(
        "gemma-3-4b",
        "google/gemma-3-4b-it",
        "Gemma 3",
        dense!("gemma3_text", 34, 2560, 10240, 8, 4, 256, 262208, true, 131072, window 1024)
    ),
    e!(
        "gemma-3-12b",
        "google/gemma-3-12b-it",
        "Gemma 3",
        dense!("gemma3_text", 48, 3840, 15360, 16, 8, 256, 262208, true, 131072, window 1024)
    ),
    e!(
        "gemma-3-27b",
        "google/gemma-3-27b-it",
        "Gemma 3",
        dense!("gemma3_text", 62, 5376, 21504, 32, 16, 128, 262208, true, 131072, window 1024)
    ),
    e!(
        "phi-3.5-mini",
        "microsoft/Phi-3.5-mini-instruct",
        "Phi",
        dense!("phi3", 32, 3072, 8192, 32, 32, 96, 32064, false, 131072)
    ),
    e!(
        "mixtral-8x7b",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mixtral (MoE)",
        || json!({
        "model_type": "mixtral", "num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000, "tie_word_embeddings": false,
        "max_position_embeddings": 32768, "num_local_experts": 8, "num_experts_per_tok": 2, "torch_dtype": "bfloat16"})
    ),
    e!(
        "qwen3-30b-a3b",
        "Qwen/Qwen3-30B-A3B",
        "Qwen 3 (MoE)",
        || json!({
        "model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048, "intermediate_size": 6144,
        "moe_intermediate_size": 768, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        "vocab_size": 151936, "tie_word_embeddings": false, "max_position_embeddings": 40960,
        "num_experts": 128, "num_experts_per_tok": 8, "torch_dtype": "bfloat16"})
    ),
];

fn norm(s: &str) -> String {
    let s = s.to_ascii_lowercase();
    let s = s.rsplit('/').next().unwrap_or(&s).to_string();
    let mut t: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '.')
        .collect();
    for suffix in [
        "instruct2501",
        "instruct2407",
        "instructv0.3",
        "instructv0.1",
        "chatv1.0",
        "instruct",
        "it",
        "chat",
    ] {
        if let Some(stripped) = t.strip_suffix(suffix) {
            t = stripped.to_string();
            break;
        }
    }
    t.replace('.', "")
}

/// Find a catalog entry by alias ("llama-3.1-8b", "Llama3.1-8B") or repo id.
pub fn lookup(name: &str) -> Option<&'static CatalogEntry> {
    let n = norm(name);
    CATALOG
        .iter()
        .find(|e| e.repo.eq_ignore_ascii_case(name))
        .or_else(|| {
            CATALOG
                .iter()
                .find(|e| norm(e.alias) == n || norm(e.repo) == n)
        })
}

/// Closest aliases for "did you mean" suggestions.
pub fn suggest(name: &str) -> Vec<&'static str> {
    let n = norm(name);
    let mut scored: Vec<(usize, &'static str)> = CATALOG
        .iter()
        .map(|e| (edit_distance(&n, &norm(e.alias)), e.alias))
        .collect();
    scored.sort();
    scored
        .into_iter()
        .take(3)
        .filter(|(d, _)| *d <= 6)
        .map(|(_, a)| a)
        .collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups() {
        assert_eq!(lookup("llama-3.1-8b").unwrap().alias, "llama-3.1-8b");
        assert_eq!(lookup("Llama3.1-8B").unwrap().alias, "llama-3.1-8b");
        assert_eq!(
            lookup("meta-llama/Llama-3.1-8B-Instruct").unwrap().alias,
            "llama-3.1-8b"
        );
        assert_eq!(lookup("gemma-2-9b-it").unwrap().alias, "gemma-2-9b");
        assert!(lookup("nope-1b").is_none());
        assert!(suggest("lama-3.1-8b").contains(&"llama-3.1-8b"));
    }

    #[test]
    fn all_entries_parse_with_sane_sizes() {
        for e in CATALOG {
            let s = e.spec();
            let gib = s.weight_bytes().as_gib();
            assert!(gib > 0.1 && gib < 1000.0, "{} {gib}", e.alias);
        }
        // Gemma 2 9B is the "19 GB" model from the original discussion.
        let g = lookup("gemma-2-9b").unwrap().spec();
        let gb = g.weight_bytes().0 as f64 / 1e9;
        assert!((gb - 18.5).abs() < 0.6, "{gb}");
        let l70 = lookup("llama-3.1-70b").unwrap().spec();
        assert!((l70.total_params() as f64 / 1e9 - 70.6).abs() < 0.5);
    }
}

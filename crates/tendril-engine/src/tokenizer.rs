//! Tokenization, chat templates and streaming detokenization.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }
}

pub struct Tok {
    inner: tokenizers::Tokenizer,
    template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    add_bos: bool,
    pub bos_id: Option<u32>,
    pub stop_ids: BTreeSet<u32>,
}

fn token_str(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

const COMMON_STOPS: &[&str] = &[
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<end_of_turn>",
    "<|end|>",
    "<|endoftext|>",
    "</s>",
    "<eos>",
];

impl Tok {
    /// Load from a model directory (tokenizer.json + tokenizer_config.json).
    pub fn from_dir(dir: &Path, config_eos: &[u32], config_bos: Option<u32>) -> Result<Tok> {
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("cannot load {}", path.display()))?;
        let tc: Value = std::fs::read(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(Value::Null);
        let mut template = match tc.get("chat_template") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Array(a)) => a
                .iter()
                .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                .or_else(|| a.first())
                .and_then(|t| t.get("template"))
                .and_then(|t| t.as_str())
                .map(String::from),
            _ => None,
        };
        if template.is_none() {
            template = std::fs::read_to_string(dir.join("chat_template.jinja")).ok();
        }
        if template.is_none() {
            if let Ok(j) = std::fs::read(dir.join("chat_template.json")) {
                if let Ok(v) = serde_json::from_slice::<Value>(&j) {
                    template = v
                        .get("chat_template")
                        .and_then(|t| t.as_str())
                        .map(String::from);
                }
            }
        }
        let bos_token = token_str(tc.get("bos_token"));
        let eos_token = token_str(tc.get("eos_token"));
        let add_bos = tc
            .get("add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let gen: Value = std::fs::read(dir.join("generation_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(Value::Null);
        let mut stop_ids: BTreeSet<u32> = config_eos.iter().copied().collect();
        match gen.get("eos_token_id") {
            Some(Value::Number(n)) => {
                stop_ids.insert(n.as_u64().unwrap_or(0) as u32);
            }
            Some(Value::Array(a)) => {
                stop_ids.extend(a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)))
            }
            _ => {}
        }
        if let Some(e) = &eos_token {
            if let Some(id) = inner.token_to_id(e) {
                stop_ids.insert(id);
            }
        }
        for s in COMMON_STOPS {
            if let Some(id) = inner.token_to_id(s) {
                stop_ids.insert(id);
            }
        }
        let bos_id = bos_token
            .as_deref()
            .and_then(|b| inner.token_to_id(b))
            .or(config_bos);
        Ok(Tok {
            inner,
            template,
            bos_token,
            eos_token,
            add_bos,
            bos_id,
            stop_ids,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Render messages with the model's chat template (ChatML if it has none).
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let tpl = self.template.clone().unwrap_or_else(|| CHATML.to_string());
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_function("strftime_now", |fmt: String| -> String {
            chrono::Local::now().format(&fmt).to_string()
        });
        env.add_template("chat", &tpl)
            .map_err(|e| anyhow!("chat template does not parse: {e}"))?;
        let t = env.get_template("chat").unwrap();
        let out = t
            .render(minijinja::context! {
                messages => messages,
                add_generation_prompt => add_generation_prompt,
                bos_token => self.bos_token.clone().unwrap_or_default(),
                eos_token => self.eos_token.clone().unwrap_or_default(),
                tools => minijinja::Value::UNDEFINED,
            })
            .map_err(|e| anyhow!("chat template failed: {e}"))?;
        Ok(out)
    }

    /// Tokenize rendered chat text (the template already contains special tokens).
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let text = self.render_chat(messages, true)?;
        let mut ids = self.encode(&text, false)?;
        let has_bos_text = self
            .bos_token
            .as_deref()
            .is_some_and(|b| !b.is_empty() && text.starts_with(b));
        if self.add_bos && !has_bos_text {
            if let Some(b) = self.bos_id {
                if ids.first() != Some(&b) {
                    ids.insert(0, b);
                }
            }
        }
        Ok(ids)
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let e = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(e.get_ids().to_vec())
    }

    /// Plain-completion encoding: honours add_bos_token.
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode(text, true)?;
        if self.add_bos {
            if let Some(b) = self.bos_id {
                if ids.first() != Some(&b) {
                    ids.insert(0, b);
                }
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow!("detokenize: {e}"))
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids.contains(&id)
    }
}

/// Emits text as tokens arrive without splitting multi-byte characters.
pub struct Detokenizer {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Default for Detokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Detokenizer {
    pub fn new() -> Self {
        Detokenizer {
            ids: Vec::new(),
            prefix: 0,
            read: 0,
        }
    }

    pub fn push(&mut self, tok: &Tok, id: u32) -> Result<String> {
        self.ids.push(id);
        let prev = tok.decode(&self.ids[self.prefix..self.read])?;
        let now = tok.decode(&self.ids[self.prefix..])?;
        if now.len() > prev.len() && !now.ends_with('\u{FFFD}') {
            let delta = now[prev.len()..].to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            Ok(delta)
        } else {
            Ok(String::new())
        }
    }
}

pub const CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

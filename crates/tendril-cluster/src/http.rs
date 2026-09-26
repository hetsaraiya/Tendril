//! OpenAI-compatible HTTP API, status endpoints and the built-in web UI.

use crate::coordinator::{Coordinator, GenOut, GenRequest, ServeError};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Stream;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use tendril_engine::sampler::SamplingParams;
use tendril_engine::tokenizer::ChatMessage;

#[derive(Clone)]
pub struct AppState {
    pub coord: Coordinator,
    pub join_command: String,
    pub model_id: String,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(|| async { "ok" }))
        .route("/api/status", get(status))
        .route("/api/events", get(events))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/tokenize", post(tokenize))
        .with_state(state)
}

async fn index() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        Html(include_str!("../web/index.html")),
    )
}

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"message": msg.into(), "type": kind, "code": status.as_u16()}})),
    )
        .into_response()
}

fn serve_err(e: ServeError) -> Response {
    match e {
        ServeError::NotReady(m) => err(StatusCode::SERVICE_UNAVAILABLE, "not_ready", m),
        ServeError::BadRequest(m) => err(StatusCode::BAD_REQUEST, "invalid_request_error", m),
        ServeError::Busy(m) => err(StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded", m),
    }
}

async fn status(State(s): State<AppState>) -> Json<Value> {
    let c = &s.coord;
    let plan = c.last_plan().and_then(|r| r.selected).map(|p| {
        json!({
            "label": p.label(),
            "tokens_per_sec": p.tokens_per_sec,
            "ttft_ms": p.ttft_ms,
            "network_ms": p.network_ms,
            "stages": p.stages.iter().map(|st| json!({
                "node": st.node_name, "components": st.describe_components(),
                "layers": [st.layer_start, st.layer_end], "peak": st.mem.peak.0, "usable": st.mem.usable.0,
                "weights": st.mem.weights.0, "kv": st.mem.kv.0, "decode_ms": st.decode_ms,
            })).collect::<Vec<_>>(),
        })
    });
    let m = c.metrics();
    let telemetry = c.telemetry().await.map(|(p, t)| {
        json!({
            "samples": t.samples,
            "step_ms": t.step_ms,
            "predicted_step_ms": p.decode_ms,
            "transfer_ms": t.transfer_ms,
            "predicted_network_ms": p.network_ms,
            "stages": p.stages.iter().enumerate().map(|(i, st)| json!({
                "node": st.node_name,
                "predicted_ms": st.decode_ms,
                "compute_ms": t.compute_ms.get(i).copied().unwrap_or(0.0),
                "queue_ms": t.queue_ms.get(i).copied().unwrap_or(0.0),
                "batch": t.batch.get(i).copied().unwrap_or(1.0),
            })).collect::<Vec<_>>(),
        })
    });
    Json(json!({
        "telemetry": telemetry,
        "model": s.model_id,
        "architecture": c.inner.spec.arch.label(),
        "params": c.inner.spec.total_params(),
        "weights": c.inner.spec.weight_bytes().0,
        "layers": c.inner.spec.num_layers,
        "context": c.inner.opts.context,
        "concurrency": c.inner.opts.concurrency,
        "format": c.inner.opts.format.label(),
        "status": c.status(),
        "plan": plan,
        "nodes": c.nodes(),
        "metrics": m,
        "events": c.history(),
        "join": s.join_command,
        "uptime_s": c.uptime().as_secs(),
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn events(
    State(s): State<AppState>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let mut rx = s.coord.subscribe();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(e) => yield Ok(SseEvent::default().data(serde_json::to_string(&e).unwrap_or_default())),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// `{"prompt": "..."}` or `{"messages": [...]}` → token count and ids.
async fn tokenize(State(s): State<AppState>, Json(body): Json<Value>) -> Response {
    let tok = &s.coord.inner.tok;
    let ids = if let Some(p) = body.get("prompt").and_then(|p| p.as_str()) {
        tok.encode_prompt(p)
    } else if let Some(m) = body.get("messages").and_then(|m| m.as_array()) {
        let msgs: Vec<ChatMessage> = m
            .iter()
            .map(|m| ChatMessage {
                role: m["role"].as_str().unwrap_or("user").into(),
                content: content_text(&m["content"]),
            })
            .collect();
        tok.encode_chat(&msgs)
    } else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "send `prompt` or `messages`",
        );
    };
    match ids {
        Ok(ids) => Json(
            json!({"count": ids.len(), "tokens": ids, "max_model_len": s.coord.inner.opts.context}),
        )
        .into_response(),
        Err(e) => err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{e:#}"),
        ),
    }
}

async fn models(State(s): State<AppState>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": s.model_id, "object": "model", "created": 0, "owned_by": "tendril"}]}),
    )
}

#[derive(Deserialize)]
struct StopField(#[serde(deserialize_with = "one_or_many")] Vec<String>);

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    let v = Value::deserialize(d)?;
    Ok(match v {
        Value::String(s) => vec![s],
        Value::Array(a) => a
            .into_iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        _ => vec![],
    })
}

#[derive(Deserialize)]
struct CommonParams {
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(default)]
    stream: bool,
    /// Extension: keep generating past end-of-sequence (for benchmarks).
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    stream_options: Option<Value>,
}

impl CommonParams {
    fn sampling(&self) -> SamplingParams {
        let d = SamplingParams::default();
        SamplingParams {
            temperature: self.temperature.unwrap_or(d.temperature).clamp(0.0, 5.0),
            top_p: self.top_p.unwrap_or(d.top_p).clamp(0.0, 1.0),
            top_k: self.top_k.unwrap_or(0),
            min_p: self.min_p.unwrap_or(0.0).clamp(0.0, 1.0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0).clamp(0.5, 3.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0).clamp(-2.0, 2.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0).clamp(-2.0, 2.0),
            seed: self.seed,
        }
    }
    fn max_tokens(&self) -> usize {
        self.max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(1024)
            .max(1)
    }
    fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}

fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                    p.get("text").and_then(|t| t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}-{:016x}", rand::random::<u64>())
}

async fn chat(State(s): State<AppState>, Json(body): Json<Value>) -> Response {
    let params: CommonParams = match serde_json::from_value(body.clone()) {
        Ok(p) => p,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid request: {e}"),
            )
        }
    };
    let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`messages` is required",
        );
    };
    let messages: Vec<ChatMessage> = msgs
        .iter()
        .map(|m| ChatMessage {
            role: m
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("user")
                .to_string(),
            content: content_text(m.get("content").unwrap_or(&Value::Null)),
        })
        .collect();
    if messages.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`messages` is empty",
        );
    }
    let prompt = match s.coord.inner.tok.encode_chat(&messages) {
        Ok(p) => p,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("{e:#}"),
            )
        }
    };
    run(s, prompt, params, true).await
}

async fn completions(State(s): State<AppState>, Json(body): Json<Value>) -> Response {
    let params: CommonParams = match serde_json::from_value(body.clone()) {
        Ok(p) => p,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid request: {e}"),
            )
        }
    };
    let text = match body.get("prompt") {
        Some(Value::String(p)) => p.clone(),
        Some(Value::Array(a)) if a.len() == 1 => a[0].as_str().unwrap_or("").to_string(),
        _ => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "`prompt` must be a string",
            )
        }
    };
    let prompt = match s.coord.inner.tok.encode_prompt(&text) {
        Ok(p) => p,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("{e:#}"),
            )
        }
    };
    run(s, prompt, params, false).await
}

async fn run(s: AppState, prompt: Vec<u32>, params: CommonParams, chat: bool) -> Response {
    let prompt_tokens = prompt.len();
    let req = GenRequest {
        prompt,
        params: params.sampling(),
        max_tokens: params.max_tokens(),
        stop: params
            .stop
            .as_ref()
            .map(|s| s.0.clone())
            .unwrap_or_default(),
        ignore_eos: params.ignore_eos,
    };
    let mut rx = match s.coord.generate(req).await {
        Ok(rx) => rx,
        Err(e) => return serve_err(e),
    };
    let id = new_id(if chat { "chatcmpl" } else { "cmpl" });
    let created = unix();
    let model = s.model_id.clone();
    let object = if chat {
        "chat.completion"
    } else {
        "text_completion"
    };

    if !params.stream {
        let mut text = String::new();
        loop {
            match rx.recv().await {
                Some(GenOut::Text(t)) => text.push_str(&t),
                Some(GenOut::Done { reason, timing }) => {
                    let choice = if chat {
                        json!({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason.openai()})
                    } else {
                        json!({"index": 0, "text": text, "finish_reason": reason.openai(), "logprobs": null})
                    };
                    return Json(json!({
                        "id": id, "object": object, "created": created, "model": model,
                        "choices": [choice],
                        "usage": {"prompt_tokens": timing.prompt_tokens, "completion_tokens": timing.completion_tokens,
                                  "total_tokens": timing.prompt_tokens + timing.completion_tokens,
                                  "prompt_tokens_details": {"cached_tokens": timing.cached_tokens}},
                        "tendril": {"ttft_ms": timing.ttft_ms, "decode_tokens_per_sec": timing.decode_tps(), "total_ms": timing.total_ms, "queue_ms": timing.queue_ms},
                    }))
                    .into_response();
                }
                Some(GenOut::Error(e)) => {
                    return err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e)
                }
                None => {
                    return err(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "server_error",
                        "generation ended unexpectedly",
                    )
                }
            }
        }
    }

    let include_usage = params.include_usage();
    let chunk_object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let stream = async_stream::stream! {
        let chunk = |delta: Value, finish: Value| -> String {
            let choice = if chat {
                json!({"index": 0, "delta": delta, "finish_reason": finish})
            } else {
                json!({"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": finish})
            };
            json!({"id": id, "object": chunk_object, "created": created, "model": model, "choices": [choice]}).to_string()
        };
        if chat {
            yield Ok::<_, Infallible>(SseEvent::default().data(chunk(json!({"role": "assistant", "content": ""}), Value::Null)));
        }
        loop {
            match rx.recv().await {
                Some(GenOut::Text(t)) => yield Ok(SseEvent::default().data(chunk(json!({"content": t}), Value::Null))),
                Some(GenOut::Done { reason, timing }) => {
                    yield Ok(SseEvent::default().data(chunk(json!({}), json!(reason.openai()))));
                    if include_usage {
                        let u = json!({"id": id, "object": chunk_object, "created": created, "model": model, "choices": [],
                            "usage": {"prompt_tokens": timing.prompt_tokens, "completion_tokens": timing.completion_tokens,
                                      "total_tokens": timing.prompt_tokens + timing.completion_tokens,
                                      "prompt_tokens_details": {"cached_tokens": timing.cached_tokens}},
                            "tendril": {"ttft_ms": timing.ttft_ms, "decode_tokens_per_sec": timing.decode_tps(), "total_ms": timing.total_ms}});
                        yield Ok(SseEvent::default().data(u.to_string()));
                    }
                    break;
                }
                Some(GenOut::Error(e)) => {
                    yield Ok(SseEvent::default().data(json!({"error": {"message": e, "type": "server_error"}}).to_string()));
                    break;
                }
                None => break,
            }
        }
        yield Ok(SseEvent::default().data("[DONE]"));
    };
    let _ = prompt_tokens;
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

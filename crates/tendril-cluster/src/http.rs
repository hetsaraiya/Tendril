//! The server's HTTP side: the dashboard, its control API (guarded by the
//! cluster token) and one OpenAI-compatible API in front of every engine.

use crate::pool::{LoadRequest, NeedsTrust, Pool, PoolChanged, Status};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Stream;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tendril_engines::{Api, Task};

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<Pool>,
    /// What another machine runs to join (shown in the dashboard).
    pub join_command: String,
    pub client: reqwest::Client,
}

impl AppState {
    pub fn new(pool: Arc<Pool>, join_command: String) -> AppState {
        AppState {
            pool,
            join_command,
            client: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("http client"),
        }
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(|| async { "ok" }))
        .route("/api/status", get(status))
        .route("/api/engines", get(engines))
        .route("/api/events", get(events))
        .route("/api/auth", get(check_auth))
        .route("/api/plan", post(plan_model))
        .route("/api/models", post(load_model))
        .route("/api/models/unload", post(unload_model))
        .route("/api/machines/remove", post(remove_machine))
        .route("/api/machines/detect", post(detect_engines))
        .route("/api/stop", post(stop_server))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/images/generations", post(images))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .layer(DefaultBodyLimit::max(64 << 20))
        .with_state(state)
}

async fn index() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        include_str!("../web/index.html"),
    )
}

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"message": msg.into(), "type": kind}})),
    )
        .into_response()
}

/// Control requests carry the cluster token: `Authorization: Bearer <token>`.
fn authorized(s: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let given = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if !given.is_empty()
        && crate::token::normalize(given) == crate::token::normalize(s.pool.token())
    {
        Ok(())
    } else {
        Err(err(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "this needs the cluster token (Authorization: Bearer <token>) — `tendril serve` prints it",
        ))
    }
}

fn load_request(body: Value) -> Result<LoadRequest, Response> {
    let r: LoadRequest = serde_json::from_value(body).map_err(|e| {
        err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            e.to_string(),
        )
    })?;
    if r.model.trim().is_empty() || r.engine.trim().is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`engine` and `model` are required",
        ));
    }
    Ok(r)
}

async fn check_auth(State(s): State<AppState>, headers: HeaderMap) -> Response {
    match authorized(&s, &headers) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(r) => r,
    }
}

/// `POST /api/plan`: what loading would do, before anything is downloaded.
async fn plan_model(
    State(s): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(r) = authorized(&s, &headers) {
        return r;
    }
    let req = match load_request(body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    match s.pool.plan(&req).await {
        Ok(p) => Json(p).into_response(),
        Err(e) => err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{e:#}"),
        ),
    }
}

async fn load_model(
    State(s): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(r) = authorized(&s, &headers) {
        return r;
    }
    let req = match load_request(body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    match s.pool.load(&req).await {
        Ok(name) => Json(json!({"loading": name})).into_response(),
        Err(e) if e.is::<PoolChanged>() => err(StatusCode::CONFLICT, "pool_changed", e.to_string()),
        Err(e) if e.is::<NeedsTrust>() => err(
            StatusCode::CONFLICT,
            "trusted_network_required",
            e.to_string(),
        ),
        Err(e) => err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{e:#}"),
        ),
    }
}

async fn unload_model(
    State(s): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(r) = authorized(&s, &headers) {
        return r;
    }
    let Some(name) = body["model"].as_str() else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`model` is required",
        );
    };
    match s.pool.unload(name) {
        Ok(n) => Json(json!({"unloaded": n})).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, "not_found", format!("{e:#}")),
    }
}

async fn remove_machine(
    State(s): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Err(r) = authorized(&s, &headers) {
        return r;
    }
    let Some(id) = body["machine"].as_str() else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`machine` is required",
        );
    };
    match s.pool.remove_machine(id) {
        Ok(n) => Json(json!({"removed": n})).into_response(),
        Err(e) => err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{e:#}"),
        ),
    }
}

/// `POST /api/machines/detect {"machine"?}`: look for engines again.
async fn detect_engines(
    State(s): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Response {
    if let Err(r) = authorized(&s, &headers) {
        return r;
    }
    let m = body
        .as_ref()
        .and_then(|b| b["machine"].as_str().map(String::from));
    s.pool.redetect(m.as_deref());
    StatusCode::ACCEPTED.into_response()
}

async fn stop_server(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = authorized(&s, &headers) {
        return r;
    }
    s.pool.request_stop();
    Json(json!({"stopping": true})).into_response()
}

fn engine_catalog() -> Vec<tendril_engines::EngineInfo> {
    tendril_engines::all().iter().map(|e| e.info()).collect()
}

async fn status(State(s): State<AppState>) -> Response {
    let deployments: Vec<Value> = s
        .pool
        .deployments()
        .iter()
        .map(|d| serde_json::to_value(d.view()).unwrap_or_default())
        .collect();
    Json(json!({
        "cluster": s.pool.name(),
        "version": env!("CARGO_PKG_VERSION"),
        "pool_version": s.pool.version(),
        "state": s.pool.state_word(),
        "machines": s.pool.machine_views(),
        "deployments": deployments,
        "engines": engine_catalog(),
        "events": s.pool.history(),
        "join": s.join_command,
    }))
    .into_response()
}

/// `GET /api/engines`: every engine and, per machine, whether it is ready.
async fn engines(State(s): State<AppState>) -> Response {
    let machines = s.pool.machine_views();
    let out: Vec<Value> = engine_catalog()
        .into_iter()
        .map(|info| {
            let per: Vec<Value> = machines
                .iter()
                .map(|m| {
                    let d = m.engines.iter().find(|d| d.engine == info.id);
                    json!({"machine": m.name, "key": m.key, "detection": d})
                })
                .collect();
            json!({"info": info, "machines": per})
        })
        .collect();
    Json(out).into_response()
}

async fn events(
    State(s): State<AppState>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let mut rx = s.pool.subscribe();
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

async fn models(State(s): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = s
        .pool
        .deployments()
        .iter()
        .map(|d| {
            json!({
                "id": d.name,
                "object": "model",
                "owned_by": d.engine.info().id,
                "created": 0,
                "task": d.model.task,
                "vision": d.model.vision,
                "ready": d.status() == Status::Ready,
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}

/// The deployment and its local API port, or the error to send.
fn route(
    s: &AppState,
    model: Option<&str>,
    task: Task,
) -> Result<(Arc<crate::pool::Deployment>, u16), Response> {
    let d = s
        .pool
        .target(model, task)
        .map_err(|m| err(StatusCode::NOT_FOUND, "model_not_found", m))?;
    match d.api_port() {
        Some(p) => {
            d.requests.fetch_add(1, Ordering::Relaxed);
            Ok((d, p))
        }
        None => Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_ready",
            match d.status() {
                Status::Failed => format!("{} failed to load — see the dashboard", d.name),
                _ => format!("{} is still loading", d.name),
            },
        )),
    }
}

/// Forward a JSON request to an OpenAI-compatible engine, streaming the
/// answer back as it comes.
async fn forward(s: &AppState, port: u16, path: &str, body: Value) -> Response {
    let url = format!("http://127.0.0.1:{port}{path}");
    let r = match s.client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                "engine_unreachable",
                format!("the engine didn't answer: {e}"),
            )
        }
    };
    let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ctype = r
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let mut resp = Response::new(Body::from_stream(r.bytes_stream()));
    *resp.status_mut() = status;
    if let Ok(v) = ctype.parse() {
        resp.headers_mut().insert(header::CONTENT_TYPE, v);
    }
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, "no-cache".parse().unwrap());
    resp
}

async fn passthrough(s: AppState, path: &str, task: Task, mut body: Value) -> Response {
    let (d, port) = match route(&s, body["model"].as_str(), task) {
        Ok(x) => x,
        Err(r) => return r,
    };
    if d.engine.api() != Api::OpenAi {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{} ({}) doesn't serve {path}", d.name, d.engine.info().name),
        );
    }
    body["model"] = json!(d.model.served_name);
    forward(&s, port, path, body).await
}

async fn chat(State(s): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(s, "/v1/chat/completions", Task::Chat, body).await
}

async fn completions(State(s): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(s, "/v1/completions", Task::Chat, body).await
}

async fn embeddings(State(s): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(s, "/v1/embeddings", Task::Chat, body).await
}

async fn images(State(s): State<AppState>, Json(mut body): Json<Value>) -> Response {
    let (d, port) = match route(&s, body["model"].as_str(), Task::ImageGeneration) {
        Ok(x) => x,
        Err(r) => return r,
    };
    if d.model.task != Task::ImageGeneration {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{} doesn't generate images", d.name),
        );
    }
    match d.engine.api() {
        Api::OpenAi => {
            body["model"] = json!(d.model.served_name);
            body.as_object_mut().map(|o| o.remove("negative_prompt"));
            forward(&s, port, "/v1/images/generations", body).await
        }
        Api::Native => {
            let base = format!("http://127.0.0.1:{port}");
            match d.engine.images(&s.client, &base, &d.model, body).await {
                Ok(v) => Json(v).into_response(),
                Err(e) => err(StatusCode::BAD_GATEWAY, "engine_error", format!("{e:#}")),
            }
        }
    }
}

/// `POST /v1/audio/transcriptions` (multipart: `file`, `model`, …).
async fn transcriptions(State(s): State<AppState>, mut mp: Multipart) -> Response {
    let mut audio = None;
    let mut fields = serde_json::Map::new();
    loop {
        match mp.next_field().await {
            Ok(Some(f)) => {
                let name = f.name().unwrap_or("").to_string();
                if name == "file" {
                    let fname = f.file_name().unwrap_or("audio.wav").to_string();
                    match f.bytes().await {
                        Ok(b) => audio = Some((b.to_vec(), fname)),
                        Err(e) => {
                            return err(
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                e.to_string(),
                            )
                        }
                    }
                } else if let Ok(t) = f.text().await {
                    fields.insert(name, json!(t));
                }
            }
            Ok(None) => break,
            Err(e) => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    e.to_string(),
                )
            }
        }
    }
    let Some((audio, fname)) = audio else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`file` is required",
        );
    };
    let model = fields
        .get("model")
        .and_then(|v| v.as_str())
        .map(String::from);
    let (d, port) = match route(&s, model.as_deref(), Task::Transcription) {
        Ok(x) => x,
        Err(r) => return r,
    };
    if d.model.task != Task::Transcription {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{} doesn't transcribe audio", d.name),
        );
    }
    let base = format!("http://127.0.0.1:{port}");
    match d
        .engine
        .transcribe(
            &s.client,
            &base,
            &d.model,
            audio,
            fname,
            Value::Object(fields),
        )
        .await
    {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(StatusCode::BAD_GATEWAY, "engine_error", format!("{e:#}")),
    }
}

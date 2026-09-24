//! Inspect a "HuggingFace" repo served by a local mock that honours HTTP
//! range requests, proving that only headers (not weights) are downloaded.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tendril_core::model::source::{inspect, resolve, InspectOptions};

fn safetensors_file(layers: usize) -> Vec<u8> {
    let mut entries = Vec::new();
    let mut off = 0u64;
    let mut add = |name: String, n: u64, entries: &mut Vec<String>| {
        entries.push(format!(r#""{name}":{{"dtype":"BF16","shape":[{n}],"data_offsets":[{off},{}]}}"#, off + n * 2));
        off += n * 2;
    };
    add("model.embed_tokens.weight".into(), 1000 * 64, &mut entries);
    for l in 0..layers {
        add(format!("model.layers.{l}.self_attn.q_proj.weight"), 64 * 64, &mut entries);
        add(format!("model.layers.{l}.mlp.up_proj.weight"), 64 * 128, &mut entries);
    }
    add("model.norm.weight".into(), 64, &mut entries);
    add("lm_head.weight".into(), 1000 * 64, &mut entries);
    let header = format!("{{{}}}", entries.join(","));
    let mut f = Vec::new();
    f.extend_from_slice(&(header.len() as u64).to_le_bytes());
    f.extend_from_slice(header.as_bytes());
    f.resize(f.len() + off as usize, 0);
    f
}

#[test]
fn inspects_remote_repo_with_range_requests() {
    let weights = Arc::new(safetensors_file(4));
    let config = r#"{"model_type":"llama","num_hidden_layers":4,"hidden_size":64,"intermediate_size":128,
        "num_attention_heads":4,"num_key_value_heads":2,"vocab_size":1000,"torch_dtype":"bfloat16"}"#;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let served = Arc::new(AtomicU64::new(0));
    let served2 = served.clone();
    let w2 = weights.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut s = stream.unwrap();
            let mut reader = BufReader::new(s.try_clone().unwrap());
            let mut req = String::new();
            reader.read_line(&mut req).unwrap();
            let mut range: Option<(u64, u64)> = None;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                if let Some(r) = h.to_ascii_lowercase().strip_prefix("range: bytes=") {
                    let (a, b) = r.trim().split_once('-').unwrap();
                    range = Some((a.parse().unwrap(), b.parse().unwrap()));
                }
            }
            let path = req.split_whitespace().nth(1).unwrap_or("").to_string();
            let (status, body): (&str, Vec<u8>) = if path.ends_with("/config.json") {
                ("200 OK", config.as_bytes().to_vec())
            } else if path.ends_with("/model.safetensors") {
                let (a, b) = range.unwrap_or((0, w2.len() as u64 - 1));
                let b = b.min(w2.len() as u64 - 1);
                ("206 Partial Content", w2[a as usize..=b as usize].to_vec())
            } else {
                ("404 Not Found", b"missing".to_vec())
            };
            served2.fetch_add(body.len() as u64, Ordering::SeqCst);
            let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = s.write_all(&body);
            let _ = s.flush();
            let mut sink = [0u8; 1];
            let _ = s.read(&mut sink);
        }
    });
    std::env::set_var("TENDRIL_HF_ENDPOINT", format!("http://{addr}"));
    std::env::set_var("XDG_CACHE_HOME", tempfile::tempdir().unwrap().keep());
    std::env::set_var("HOME", tempfile::tempdir().unwrap().keep());
    let r = resolve("mock-org/tiny-llama").unwrap();
    let spec = inspect(&r, &InspectOptions::default()).unwrap();
    assert!(spec.bytes_measured, "{:?}", spec.notes);
    assert_eq!(spec.bytes.embed.0, 1000 * 64 * 2);
    assert_eq!(spec.bytes.layers.len(), 4);
    assert_eq!(spec.bytes.layers[0].0, (64 * 64 + 64 * 128) * 2);
    assert!(!spec.tie_embeddings);
    // Only the config and the safetensors header crossed the wire.
    let total = served.load(Ordering::SeqCst);
    assert!(total < (weights.len() as u64) / 10, "downloaded {total} of {} bytes", weights.len());
}

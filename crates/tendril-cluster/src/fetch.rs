//! Downloading exactly the tensors one stage needs from a remote checkpoint
//! (HuggingFace or a mirror) with HTTP range requests.

use crate::proto::{TensorEntry, WeightSource};
use crate::shard::ShardWriter;
use anyhow::{bail, Context, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tendril_core::model::source::{download_client, hf_endpoint};

/// A client for `base_url`. The machine's own HuggingFace token is attached
/// only when the URL is its configured HuggingFace endpoint, so a coordinator
/// can never make it send the token elsewhere.
pub fn client_for(base_url: &str) -> Result<reqwest::blocking::Client> {
    if !(base_url.starts_with("https://") || base_url.starts_with("http://")) {
        bail!("refusing to download weights from {base_url}");
    }
    if base_url.starts_with(&format!("{}/", hf_endpoint())) {
        return download_client();
    }
    Ok(reqwest::blocking::Client::builder()
        .user_agent(concat!("tendril/", env!("CARGO_PKG_VERSION")))
        .timeout(None)
        .build()?)
}

/// Bytes per range request. Large tensors are split so several connections
/// share them; small ones are fetched whole.
const CHUNK: u64 = 32 << 20;

/// Connections per machine (`TENDRIL_DOWNLOAD_STREAMS` overrides).
fn streams() -> usize {
    std::env::var("TENDRIL_DOWNLOAD_STREAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(8)
        .min(64)
}

/// One range request: `len` bytes of `file` from `src`, into the shard at `dst`.
struct Piece {
    file: String,
    src: u64,
    dst: u64,
    len: u64,
    tensor: String,
}

/// Download a stage's tensors into a shard at `path`, over several
/// connections at once (each tensor split into ≤32 MiB range requests,
/// each retried on its own). `progress(done, total)` is called as bytes
/// arrive.
pub fn download_shard(
    client: &reqwest::blocking::Client,
    source: &WeightSource,
    entries: &[TensorEntry],
    path: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf> {
    if entries.len() != source.files.len() {
        bail!("weight source doesn't match the tensor list");
    }
    let mut w = ShardWriter::create(path, entries)?;
    let total = w.total;
    let base = source.base_url.trim_end_matches('/');
    let mut pieces = Vec::new();
    for (e, (file, offset)) in entries.iter().zip(&source.files) {
        if file.is_empty() || file.contains("..") || file.starts_with('/') {
            bail!("invalid checkpoint file name {file:?}");
        }
        let (dst, len) = w.region(&e.name).context("tensor missing from shard")?;
        let mut at = 0;
        while at < len {
            let n = CHUNK.min(len - at);
            pieces.push(Piece {
                file: file.clone(),
                src: offset.checked_add(at).context("bad offset")?,
                dst: dst + at,
                len: n,
                tensor: e.name.clone(),
            });
            at += n;
        }
    }
    let queue = Mutex::new(pieces.into_iter());
    let done = AtomicU64::new(0);
    let failed: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let workers = streams();
    std::thread::scope(|s| -> Result<()> {
        let mut handles = Vec::new();
        for _ in 0..workers {
            let (queue, done, failed) = (&queue, &done, &failed);
            let shard = w.path().to_path_buf();
            handles.push(s.spawn(move || {
                let r = (|| -> Result<()> {
                    let mut out = std::fs::OpenOptions::new().write(true).open(&shard)?;
                    let mut buf = vec![0u8; 1 << 20];
                    loop {
                        if failed.lock().unwrap().is_some() {
                            return Ok(());
                        }
                        let Some(p) = queue.lock().unwrap().next() else {
                            return Ok(());
                        };
                        let url = format!("{base}/{}", p.file);
                        let mut tries = 0;
                        loop {
                            let mut got = 0u64;
                            match fetch_piece(client, &url, &p, &mut out, &mut buf, &mut |n| {
                                got += n;
                                done.fetch_add(n, Ordering::Relaxed);
                            }) {
                                Ok(()) => break,
                                Err(err) => {
                                    done.fetch_sub(got, Ordering::Relaxed);
                                    let fatal = format!("{err}").contains("access denied");
                                    if fatal || tries >= 3 {
                                        return Err(err);
                                    }
                                    tries += 1;
                                    tracing::warn!("retrying {}: {err:#}", p.tensor);
                                    std::thread::sleep(std::time::Duration::from_millis(
                                        300 << tries,
                                    ));
                                }
                            }
                        }
                    }
                })();
                if let Err(e) = r {
                    failed.lock().unwrap().get_or_insert(e);
                }
            }));
        }
        // Report progress from this thread while the workers download.
        while handles.iter().any(|h| !h.is_finished()) {
            progress(done.load(Ordering::Relaxed), total);
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        Ok(())
    })?;
    if let Some(e) = failed.into_inner().unwrap() {
        return Err(e);
    }
    let got = done.load(Ordering::Relaxed);
    progress(got, total);
    w.add_written(got);
    w.finish()
}

/// Fetch one piece into `out` at its place in the shard.
fn fetch_piece(
    client: &reqwest::blocking::Client,
    url: &str,
    p: &Piece,
    out: &mut std::fs::File,
    buf: &mut [u8],
    got: &mut dyn FnMut(u64),
) -> Result<()> {
    if p.len == 0 {
        return Ok(());
    }
    let end = p.src.checked_add(p.len - 1).context("bad offset")?;
    let mut resp = client
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes={}-{end}", p.src))
        .send()
        .with_context(|| format!("GET {url}"))?;
    match resp.status().as_u16() {
        206 => {}
        401 | 403 => bail!("{url}: access denied — the model may be gated: accept its license on huggingface.co and set HF_TOKEN on this machine"),
        s => bail!("{url}: HTTP {s} (expected a byte range)"),
    }
    out.seek(SeekFrom::Start(p.dst))?;
    let mut at = 0u64;
    while at < p.len {
        let want = buf.len().min((p.len - at) as usize);
        let n = resp.read(&mut buf[..want])?;
        if n == 0 {
            bail!("connection closed early");
        }
        out.write_all(&buf[..n])?;
        at += n as u64;
        got(n as u64);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// A range server that serves requests concurrently, fails the first
    /// one, and records how many requests were in flight at once.
    fn server(file: Arc<Vec<u8>>) -> (String, Arc<AtomicUsize>) {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/org/m/resolve/main", l.local_addr().unwrap());
        let (active, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let first = Arc::new(AtomicUsize::new(0));
        let peak2 = peak.clone();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let mut s = s.unwrap();
                let (file, active, peak, first) =
                    (file.clone(), active.clone(), peak2.clone(), first.clone());
                std::thread::spawn(move || {
                    let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(n, Ordering::SeqCst);
                    let mut r = BufReader::new(s.try_clone().unwrap());
                    let (mut line, mut range) = (String::new(), (0usize, 0usize));
                    r.read_line(&mut line).unwrap();
                    loop {
                        let mut h = String::new();
                        r.read_line(&mut h).unwrap();
                        if h.trim().is_empty() {
                            break;
                        }
                        if let Some(v) = h.to_ascii_lowercase().strip_prefix("range: bytes=") {
                            let (a, b) = v.trim().split_once('-').unwrap();
                            range = (a.parse().unwrap(), b.parse().unwrap());
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(30));
                    if first.fetch_add(1, Ordering::SeqCst) == 0 {
                        let _ = write!(
                            s,
                            "HTTP/1.1 503 Busy\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        );
                    } else {
                        let body = &file[range.0..=range.1];
                        let _ = write!(
                            s,
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = s.write_all(body);
                    }
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        (base, peak)
    }

    #[test]
    fn downloads_in_parallel_chunks_and_retries() {
        // A checkpoint with a 70 MiB tensor (split into 32 MiB requests) and a small one.
        let big = 70usize << 20;
        let small = 5000usize;
        let file: Vec<u8> = (0..big + small + 300)
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        let (base, peak) = server(Arc::new(file.clone()));
        let entries = vec![
            TensorEntry {
                name: "a".into(),
                dtype: "U8".into(),
                shape: vec![big],
                len: big as u64,
            },
            TensorEntry {
                name: "b".into(),
                dtype: "U8".into(),
                shape: vec![small],
                len: small as u64,
            },
        ];
        let source = WeightSource {
            base_url: base.clone(),
            files: vec![
                ("m.safetensors".into(), 100),
                ("m.safetensors".into(), (big + 200) as u64),
            ],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.safetensors");
        let mut last = 0;
        let out = download_shard(
            &client_for(&base).unwrap(),
            &source,
            &entries,
            &path,
            &mut |d, t| {
                assert!(d <= t);
                last = d;
            },
        )
        .unwrap();
        assert_eq!(last, (big + small) as u64);
        assert!(peak.load(Ordering::SeqCst) > 1, "requests ran in parallel");
        // The shard holds exactly the two tensors' bytes, in order.
        let got = std::fs::read(&out).unwrap();
        let hlen = u64::from_le_bytes(got[..8].try_into().unwrap()) as usize;
        let data = &got[8 + hlen..];
        assert_eq!(&data[..big], &file[100..100 + big]);
        assert_eq!(&data[big..], &file[big + 200..big + 200 + small]);
    }
}

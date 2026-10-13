//! Downloading model files: several HTTP range requests at once per file,
//! each retried on its own, into a `.part` file renamed when complete.

use anyhow::{bail, Context, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tendril_core::model::source::{download_client, hf_endpoint};

/// A client for `url`. The machine's own HuggingFace token is attached only
/// when the URL is its configured HuggingFace endpoint, so a server can never
/// make it send the token elsewhere.
pub fn client_for(url: &str) -> Result<reqwest::blocking::Client> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        bail!("refusing to download from {url}");
    }
    if url.starts_with(&format!("{}/", hf_endpoint())) {
        return download_client();
    }
    Ok(reqwest::blocking::Client::builder()
        .user_agent(concat!("tendril/", env!("CARGO_PKG_VERSION")))
        .timeout(None)
        .build()?)
}

/// Bytes per range request.
const CHUNK: u64 = 32 << 20;

/// Connections per file (`TENDRIL_DOWNLOAD_STREAMS` overrides).
fn streams() -> usize {
    std::env::var("TENDRIL_DOWNLOAD_STREAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(8)
        .min(64)
}

/// One range request: `len` bytes at `at`.
struct Piece {
    at: u64,
    len: u64,
}

/// Where the URL a cached file came from is recorded.
fn source_path(dest: &Path) -> std::path::PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".source");
    s.into()
}

/// Download `url` (`size` bytes) to `dest`. A complete file of the right size
/// downloaded from the same URL (which names the revision) is kept as is.
/// `progress(done, total)` is called as bytes arrive.
pub fn download_file(
    url: &str,
    size: u64,
    dest: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<()> {
    let source = source_path(dest);
    let same_source = std::fs::read_to_string(&source).is_ok_and(|s| s == url);
    if same_source && std::fs::metadata(dest).is_ok_and(|m| m.len() == size && size > 0) {
        progress(size, size);
        return Ok(());
    }
    let _ = std::fs::remove_file(&source);
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let client = client_for(url)?;
    let part = dest.with_extension(format!(
        "{}part",
        dest.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));
    if size == 0 {
        // Unknown size: one plain request.
        let mut resp = client
            .get(url)
            .send()
            .with_context(|| format!("GET {url}"))?;
        check(resp.status().as_u16(), url, false)?;
        let mut out = std::fs::File::create(&part)?;
        let n = std::io::copy(&mut resp, &mut out)?;
        progress(n, n);
        drop(out);
        std::fs::rename(&part, dest)?;
        std::fs::write(&source, url)?;
        return Ok(());
    }
    {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&part)?;
        f.set_len(size)?;
    }
    let mut pieces = Vec::new();
    let mut at = 0;
    while at < size {
        let n = CHUNK.min(size - at);
        pieces.push(Piece { at, len: n });
        at += n;
    }
    let queue = Mutex::new(pieces.into_iter());
    let done = AtomicU64::new(0);
    let failed: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..streams() {
            let (queue, done, failed, client, part) = (&queue, &done, &failed, &client, &part);
            handles.push(s.spawn(move || {
                let r = (|| -> Result<()> {
                    let mut out = std::fs::OpenOptions::new().write(true).open(part)?;
                    let mut buf = vec![0u8; 1 << 20];
                    loop {
                        if failed.lock().unwrap().is_some() {
                            return Ok(());
                        }
                        let Some(p) = queue.lock().unwrap().next() else {
                            return Ok(());
                        };
                        let mut tries = 0;
                        loop {
                            let mut got = 0u64;
                            match fetch_piece(client, url, &p, &mut out, &mut buf, &mut |n| {
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
                                    tracing::warn!("retrying {url} @{}: {err:#}", p.at);
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
        while handles.iter().any(|h| !h.is_finished()) {
            progress(done.load(Ordering::Relaxed), size);
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    });
    if let Some(e) = failed.into_inner().unwrap() {
        return Err(e);
    }
    progress(done.load(Ordering::Relaxed), size);
    std::fs::rename(&part, dest).with_context(|| format!("save {}", dest.display()))?;
    std::fs::write(&source, url)?;
    Ok(())
}

fn check(status: u16, url: &str, ranged: bool) -> Result<()> {
    match status {
        206 if ranged => Ok(()),
        200 if !ranged => Ok(()),
        401 | 403 => bail!("{url}: access denied — the model may be gated: accept its license on huggingface.co and set HF_TOKEN on this machine"),
        s if ranged => bail!("{url}: HTTP {s} (expected a byte range)"),
        s => bail!("{url}: HTTP {s}"),
    }
}

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
    let end = p.at + p.len - 1;
    let mut resp = client
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes={}-{end}", p.at))
        .send()
        .with_context(|| format!("GET {url}"))?;
    check(resp.status().as_u16(), url, true)?;
    out.seek(SeekFrom::Start(p.at))?;
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
        let big = 70usize << 20;
        let file: Vec<u8> = (0..big + 777).map(|i| (i * 31 % 251) as u8).collect();
        let (base, peak) = server(Arc::new(file.clone()));
        let url = format!("{base}/m.gguf");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("org/m.gguf");
        let mut last = 0;
        download_file(&url, file.len() as u64, &dest, &mut |d, t| {
            assert!(d <= t);
            last = d;
        })
        .unwrap();
        assert_eq!(last, file.len() as u64);
        assert!(peak.load(Ordering::SeqCst) > 1, "requests ran in parallel");
        assert_eq!(std::fs::read(&dest).unwrap(), file);
        assert!(!dir.path().join("org/m.gguf.part").exists());
        // Already there from the same URL: nothing is fetched again.
        download_file(&url, file.len() as u64, &dest, &mut |_, _| {}).unwrap();
        // Same name and size from another URL (another revision): fetched again.
        let other = download_file(
            "http://127.0.0.1:1/other",
            file.len() as u64,
            &dest,
            &mut |_, _| {},
        );
        assert!(other.is_err());
    }
}

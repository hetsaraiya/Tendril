//! TCP over Tendril's encrypted links.
//!
//! Engines listen on 127.0.0.1 only. When something on one machine needs an
//! engine port on another (the server reaching a model's API, llama-server
//! reaching a ggml-rpc-server), it connects to a local [`Forwarder`], which
//! opens an authenticated, encrypted connection to the other machine's tunnel
//! listener and names the role it wants. The listener only forwards to ports
//! of roles it is running ([`Targets`]); everything else is refused.

use crate::proto::Msg;
use crate::wire;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

/// Ports this machine lets tunnels reach, by `deployment/role`.
#[derive(Clone, Default)]
pub struct Targets(Arc<Mutex<HashMap<String, u16>>>);

impl Targets {
    pub fn insert(&self, key: &str, port: u16) {
        self.0.lock().unwrap().insert(key.to_string(), port);
    }

    pub fn remove(&self, key: &str) {
        self.0.lock().unwrap().remove(key);
    }

    pub fn get(&self, key: &str) -> Option<u16> {
        self.0.lock().unwrap().get(key).copied()
    }
}

/// Accept tunnels on `listener` forever.
pub async fn serve(listener: TcpListener, psk: [u8; 32], targets: Targets) {
    loop {
        let Ok((s, _)) = listener.accept().await else {
            continue;
        };
        let targets = targets.clone();
        tokio::spawn(async move {
            if let Err(e) = accept_one(s, &psk, &targets).await {
                tracing::debug!("tunnel: {e:#}");
            }
        });
    }
}

async fn accept_one(s: TcpStream, psk: &[u8; 32], targets: &Targets) -> Result<()> {
    let mut c = wire::accept(s, psk).await?;
    let target = match c.reader.recv().await? {
        Msg::Open { target } => target,
        _ => bail!("expected Open"),
    };
    let Some(port) = targets.get(&target) else {
        c.writer
            .send(&Msg::Reject {
                reason: format!("{target} is not running here"),
            })
            .await?;
        bail!("refused tunnel to {target}");
    };
    let local = match TcpStream::connect(("127.0.0.1", port)).await {
        Ok(l) => l,
        Err(e) => {
            c.writer
                .send(&Msg::Reject {
                    reason: format!("{target} is not accepting connections: {e}"),
                })
                .await?;
            bail!("{target}: {e}");
        }
    };
    local.set_nodelay(true)?;
    c.writer.send(&Msg::Opened).await?;
    pipe(c, local).await
}

/// Open a tunnel to `target` on the machine whose tunnel listener is `addr`.
pub async fn open(addr: &str, psk: &[u8; 32], target: &str) -> Result<wire::Conn> {
    let mut c = wire::connect(addr, psk).await?;
    c.writer
        .send(&Msg::Open {
            target: target.into(),
        })
        .await?;
    match c.reader.recv().await? {
        Msg::Opened => Ok(c),
        Msg::Reject { reason } => bail!("{reason}"),
        other => bail!("unexpected tunnel reply {other:?}"),
    }
}

/// Copy bytes both ways between an encrypted link and a local socket until
/// both sides are done.
pub async fn pipe(c: wire::Conn, local: TcpStream) -> Result<()> {
    let (mut lr, mut lw) = local.into_split();
    let (mut rr, mut rw) = (c.reader, c.writer);
    let up = async move {
        let mut buf = vec![0u8; 256 << 10];
        loop {
            let n = lr.read(&mut buf).await?;
            rw.send_bytes(&buf[..n]).await?;
            if n == 0 {
                return anyhow::Ok(());
            }
        }
    };
    let down = async move {
        loop {
            let b = rr.recv_bytes().await?;
            if b.is_empty() {
                lw.shutdown().await.ok();
                return anyhow::Ok(());
            }
            lw.write_all(&b).await?;
        }
    };
    let (a, b) = tokio::join!(up, down);
    a.and(b)
}

/// Where a forwarder sends connections.
#[derive(Clone, Debug)]
pub enum Route {
    /// A port on this machine (no encryption needed).
    Local(u16),
    /// A role on another machine, through its tunnel listener.
    Remote { tunnel: String, target: String },
}

/// A 127.0.0.1 listener that sends every connection along a [`Route`].
pub struct Forwarder {
    pub port: u16,
    task: JoinHandle<()>,
}

impl Forwarder {
    pub async fn start(route: Route, psk: [u8; 32]) -> Result<Forwarder> {
        if let Route::Local(port) = route {
            // Nothing to wrap: hand out the port itself.
            return Ok(Forwarder {
                port,
                task: tokio::spawn(async {}),
            });
        }
        let l = TcpListener::bind(("127.0.0.1", 0))
            .await
            .context("cannot open a local port for a tunnel")?;
        let port = l.local_addr()?.port();
        let task = tokio::spawn(async move {
            loop {
                let Ok((s, _)) = l.accept().await else {
                    continue;
                };
                let route = route.clone();
                tokio::spawn(async move {
                    let _ = s.set_nodelay(true);
                    if let Route::Remote { tunnel, target } = &route {
                        match open(tunnel, &psk, target).await {
                            Ok(c) => {
                                let _ = pipe(c, s).await;
                            }
                            Err(e) => tracing::warn!("tunnel to {target} via {tunnel}: {e:#}"),
                        }
                    }
                });
            }
        });
        Ok(Forwarder { port, task })
    }
}

impl Drop for Forwarder {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bytes_cross_the_tunnel_only_to_allowed_targets() {
        let psk = crate::token::psk("TEST-TOKEN");
        // An "engine": echoes in upper case until EOF.
        let engine = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let eport = engine.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (mut s, _) = engine.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    s.read_to_end(&mut buf).await.unwrap();
                    s.write_all(&buf.to_ascii_uppercase()).await.unwrap();
                });
            }
        });
        let targets = Targets::default();
        targets.insert("d1/server", eport);
        let tl = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let taddr = tl.local_addr().unwrap().to_string();
        tokio::spawn(serve(tl, psk, targets.clone()));

        let f = Forwarder::start(
            Route::Remote {
                tunnel: taddr.clone(),
                target: "d1/server".into(),
            },
            psk,
        )
        .await
        .unwrap();
        let mut s = TcpStream::connect(("127.0.0.1", f.port)).await.unwrap();
        let big: Vec<u8> = (0..3_000_000u32).map(|i| b'a' + (i % 26) as u8).collect();
        s.write_all(&big).await.unwrap();
        s.shutdown().await.unwrap();
        let mut back = Vec::new();
        s.read_to_end(&mut back).await.unwrap();
        assert_eq!(back, big.to_ascii_uppercase());

        // Unknown roles and wrong tokens get nothing.
        assert!(open(&taddr, &psk, "d1/other").await.is_err());
        assert!(open(&taddr, &crate::token::psk("WRONG"), "d1/server")
            .await
            .is_err());
        targets.remove("d1/server");
        assert!(open(&taddr, &psk, "d1/server").await.is_err());
    }
}

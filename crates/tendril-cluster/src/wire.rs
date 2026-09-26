//! Authenticated, encrypted, framed connections (Noise NNpsk0 over TCP).
//!
//! Every connection proves knowledge of the cluster token before any data
//! flows; a machine without the token cannot join, read activations or
//! inject work. Messages are length-prefixed and bounded before allocation.

use crate::proto::Msg;
use anyhow::{bail, Context, Result};
use snow::StatelessTransportState;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;

const PATTERN: &str = "Noise_NNpsk0_25519_ChaChaPoly_BLAKE2s";
const MAGIC: &[u8; 6] = b"TNDRL\x01";
const MAX_CHUNK: usize = 65535 - 16;
/// Largest message accepted (a prefill chunk of a very large model fits easily).
pub const MAX_MESSAGE: usize = 512 << 20;

pub struct Reader {
    inner: BufReader<OwnedReadHalf>,
    st: Arc<StatelessTransportState>,
    nonce: u64,
    ct: Vec<u8>,
}

pub struct Writer {
    inner: BufWriter<OwnedWriteHalf>,
    st: Arc<StatelessTransportState>,
    nonce: u64,
    ct: Vec<u8>,
}

pub struct Conn {
    pub reader: Reader,
    pub writer: Writer,
    pub peer: SocketAddr,
}

async fn write_hs(s: &mut TcpStream, data: &[u8]) -> Result<()> {
    s.write_all(&(data.len() as u16).to_be_bytes()).await?;
    s.write_all(data).await?;
    Ok(())
}

async fn read_hs(s: &mut TcpStream) -> Result<Vec<u8>> {
    let mut l = [0u8; 2];
    s.read_exact(&mut l).await?;
    let mut b = vec![0u8; u16::from_be_bytes(l) as usize];
    s.read_exact(&mut b).await?;
    Ok(b)
}

fn split(stream: TcpStream, st: StatelessTransportState, peer: SocketAddr) -> Conn {
    let st = Arc::new(st);
    let (r, w) = stream.into_split();
    Conn {
        reader: Reader {
            inner: BufReader::with_capacity(256 << 10, r),
            st: st.clone(),
            nonce: 0,
            ct: vec![0u8; 65535],
        },
        writer: Writer {
            inner: BufWriter::with_capacity(256 << 10, w),
            st,
            nonce: 0,
            ct: vec![0u8; 65535],
        },
        peer,
    }
}

/// Connect and authenticate as the initiator.
pub async fn connect(addr: &str, psk: &[u8; 32]) -> Result<Conn> {
    let mut s = tokio::time::timeout(Duration::from_secs(10), TcpStream::connect(addr))
        .await
        .with_context(|| format!("timed out connecting to {addr}"))?
        .with_context(|| format!("cannot connect to {addr}"))?;
    s.set_nodelay(true)?;
    let peer = s.peer_addr()?;
    s.write_all(MAGIC).await?;
    let mut hs = snow::Builder::new(PATTERN.parse()?)
        .psk(0, psk)?
        .build_initiator()?;
    let mut buf = vec![0u8; 1024];
    let n = hs.write_message(&[], &mut buf)?;
    write_hs(&mut s, &buf[..n]).await?;
    let msg = tokio::time::timeout(Duration::from_secs(10), read_hs(&mut s))
        .await
        .context("handshake timed out")?
        .context("connection closed during handshake (is the token right?)")?;
    hs.read_message(&msg, &mut buf)
        .map_err(|_| anyhow::anyhow!("authentication failed: wrong cluster token"))?;
    Ok(split(s, hs.into_stateless_transport_mode()?, peer))
}

/// Accept and authenticate as the responder.
pub async fn accept(mut s: TcpStream, psk: &[u8; 32]) -> Result<Conn> {
    s.set_nodelay(true)?;
    let peer = s.peer_addr()?;
    let mut magic = [0u8; 6];
    tokio::time::timeout(Duration::from_secs(10), s.read_exact(&mut magic))
        .await
        .context("handshake timed out")??;
    if &magic != MAGIC {
        bail!("{peer} is not a Tendril peer (or speaks a different protocol version)");
    }
    let mut hs = snow::Builder::new(PATTERN.parse()?)
        .psk(0, psk)?
        .build_responder()?;
    let mut buf = vec![0u8; 1024];
    let msg = tokio::time::timeout(Duration::from_secs(10), read_hs(&mut s))
        .await
        .context("handshake timed out")??;
    hs.read_message(&msg, &mut buf)
        .map_err(|_| anyhow::anyhow!("{peer} failed authentication (wrong cluster token)"))?;
    let n = hs.write_message(&[], &mut buf)?;
    write_hs(&mut s, &buf[..n]).await?;
    Ok(split(s, hs.into_stateless_transport_mode()?, peer))
}

impl Writer {
    pub async fn send(&mut self, msg: &Msg) -> Result<()> {
        let bytes = postcard::to_allocvec(msg)?;
        self.send_raw(&bytes).await?;
        self.inner.flush().await?;
        Ok(())
    }

    async fn send_raw(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > MAX_MESSAGE {
            bail!(
                "message of {} bytes exceeds the {} byte limit",
                bytes.len(),
                MAX_MESSAGE
            );
        }
        self.inner
            .write_all(&(bytes.len() as u32).to_be_bytes())
            .await?;
        for chunk in bytes.chunks(MAX_CHUNK) {
            let n = self.st.write_message(self.nonce, chunk, &mut self.ct)?;
            self.nonce += 1;
            self.inner.write_all(&(n as u16).to_be_bytes()).await?;
            self.inner.write_all(&self.ct[..n]).await?;
        }
        if bytes.is_empty() {
            // Keep framing symmetric: one empty encrypted chunk.
            let n = self.st.write_message(self.nonce, &[], &mut self.ct)?;
            self.nonce += 1;
            self.inner.write_all(&(n as u16).to_be_bytes()).await?;
            self.inner.write_all(&self.ct[..n]).await?;
        }
        Ok(())
    }
}

impl Reader {
    pub async fn recv(&mut self) -> Result<Msg> {
        let mut l = [0u8; 4];
        self.inner.read_exact(&mut l).await?;
        let total = u32::from_be_bytes(l) as usize;
        if total > MAX_MESSAGE {
            bail!("peer announced a {total}-byte message (limit {MAX_MESSAGE})");
        }
        let mut out = Vec::with_capacity(total);
        let mut pt = vec![0u8; 65535];
        loop {
            let mut cl = [0u8; 2];
            self.inner.read_exact(&mut cl).await?;
            let n = u16::from_be_bytes(cl) as usize;
            if n < 16 {
                bail!("malformed frame");
            }
            self.inner.read_exact(&mut self.ct[..n]).await?;
            let m = self
                .st
                .read_message(self.nonce, &self.ct[..n], &mut pt)
                .map_err(|_| anyhow::anyhow!("message failed authentication"))?;
            self.nonce += 1;
            out.extend_from_slice(&pt[..m]);
            if out.len() >= total {
                break;
            }
        }
        if out.len() != total {
            bail!("frame length mismatch");
        }
        postcard::from_bytes(&out).context("malformed message")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{Payload, WireTensor};

    #[tokio::test]
    async fn roundtrip_and_wrong_token() {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let good = crate::token::psk("AAAA-BBBB");
        let server = tokio::spawn(async move {
            let (s, _) = l.accept().await.unwrap();
            let mut c = accept(s, &good).await.unwrap();
            for _ in 0..2 {
                let m = c.reader.recv().await.unwrap();
                c.writer.send(&m).await.unwrap();
            }
            // Second client uses the wrong token.
            let (s, _) = l.accept().await.unwrap();
            assert!(accept(s, &good).await.is_err());
        });
        let mut c = connect(&addr, &good).await.unwrap();
        c.writer.send(&Msg::Ping { t: 42 }).await.unwrap();
        assert!(matches!(
            c.reader.recv().await.unwrap(),
            Msg::Ping { t: 42 }
        ));
        // A large message spans many encrypted chunks.
        let big = Msg::Forward {
            epoch: 1,
            seq: 2,
            pos: 3,
            payload: Payload::Hidden(WireTensor {
                dtype: "f32".into(),
                shape: vec![1, 1000, 1000],
                data: vec![7u8; 4_000_000],
            }),
            want_logits: true,
            sample: None,
            trace: vec![],
            draft: None,
        };
        c.writer.send(&big).await.unwrap();
        match c.reader.recv().await.unwrap() {
            Msg::Forward {
                payload: Payload::Hidden(t),
                ..
            } => assert_eq!(t.data.len(), 4_000_000),
            _ => panic!(),
        }
        let bad = crate::token::psk("WRONG");
        assert!(connect(&addr, &bad).await.is_err());
        server.await.unwrap();
    }
}

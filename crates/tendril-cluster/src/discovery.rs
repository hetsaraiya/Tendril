//! Zero-config LAN discovery.
//!
//! A coordinator broadcasts a small beacon every second. It names the model,
//! the control port and a *fingerprint* of the cluster token (a hash, never
//! the token itself), so `tendril join --token T` can find the right cluster
//! without an address, and strangers learn nothing that lets them join.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;

pub const DISCOVERY_PORT: u16 = 7419;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Beacon {
    /// Always "tendril" (filters unrelated traffic).
    pub app: String,
    pub protocol: u32,
    pub version: String,
    pub model: String,
    pub host: String,
    pub control_port: u16,
    pub fingerprint: String,
    pub machines: usize,
    pub state: String,
}

/// Public fingerprint of a token: identifies a cluster without revealing it.
pub fn fingerprint(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(b"tendril-cluster-fingerprint-v1:");
    h.update(crate::token::normalize(token).as_bytes());
    h.finalize()[..6]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn std_socket(port: u16) -> std::io::Result<std::net::UdpSocket> {
    use socket2::{Domain, Protocol, Socket, Type};
    let s = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    s.set_reuse_address(true)?;
    #[cfg(all(unix, not(target_os = "solaris")))]
    s.set_reuse_port(true)?;
    s.set_broadcast(true)?;
    s.set_nonblocking(true)?;
    s.bind(&SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port).into())?;
    Ok(s.into())
}

/// Broadcast `beacon()` every second until the task is dropped.
pub async fn announce(port: u16, beacon: impl Fn() -> Beacon) {
    let Ok(sock) = std_socket(0).and_then(UdpSocket::from_std) else {
        tracing::warn!("LAN discovery unavailable (cannot open a UDP socket)");
        return;
    };
    let targets = [
        SocketAddr::from((Ipv4Addr::BROADCAST, port)),
        SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
    ];
    loop {
        if let Ok(bytes) = serde_json::to_vec(&beacon()) {
            for t in &targets {
                let _ = sock.send_to(&bytes, t).await;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// Listen for beacons for `wait`; one entry per cluster (address, beacon).
pub async fn discover(port: u16, wait: Duration) -> std::io::Result<Vec<(SocketAddr, Beacon)>> {
    let sock = UdpSocket::from_std(std_socket(port)?)?;
    let mut found: BTreeMap<String, (SocketAddr, Beacon)> = BTreeMap::new();
    let deadline = Instant::now() + wait;
    let mut buf = vec![0u8; 4096];
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        match tokio::time::timeout(left, sock.recv_from(&mut buf)).await {
            Ok(Ok((n, from))) => {
                if let Ok(b) = serde_json::from_slice::<Beacon>(&buf[..n]) {
                    if b.app == "tendril" {
                        let addr = SocketAddr::new(from.ip(), b.control_port);
                        found.insert(format!("{}#{}", b.fingerprint, addr), (addr, b));
                    }
                }
            }
            Ok(Err(e)) => return Err(e),
            Err(_) => break,
        }
    }
    // Prefer a LAN address over loopback when the same cluster answered twice.
    let mut out: Vec<(SocketAddr, Beacon)> = Vec::new();
    for (_, (addr, b)) in found {
        if let Some(existing) = out.iter_mut().find(|(a, e)| {
            e.fingerprint == b.fingerprint
                && e.control_port == b.control_port
                && (a.ip().is_loopback() || addr.ip().is_loopback())
        }) {
            if existing.0.ip().is_loopback() && !addr.ip().is_loopback() {
                *existing = (addr, b);
            }
            continue;
        }
        out.push((addr, b));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprints_hide_the_token() {
        let t = "7Q2K-9XMP-4HVD-J3FA";
        assert_eq!(fingerprint(t), fingerprint("7q2k 9xmp 4hvd j3fa"));
        assert_ne!(fingerprint(t), fingerprint("7Q2K-9XMP-4HVD-J3FB"));
        assert!(!fingerprint(t).contains("7Q2K"));
    }

    #[tokio::test]
    async fn beacons_are_found() {
        let port = 47000 + (std::process::id() % 2000) as u16;
        let b = Beacon {
            app: "tendril".into(),
            protocol: 1,
            version: "t".into(),
            model: "m".into(),
            host: "h".into(),
            control_port: 7420,
            fingerprint: fingerprint("AAAA"),
            machines: 1,
            state: "ready".into(),
        };
        let b2 = b.clone();
        let task = tokio::spawn(async move { announce(port, move || b2.clone()).await });
        let found = discover(port, Duration::from_millis(2500)).await.unwrap();
        task.abort();
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].1, b);
        assert_eq!(found[0].0.port(), 7420);
    }
}

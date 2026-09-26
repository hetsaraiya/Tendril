//! Cluster join tokens: short, typeable secrets that authenticate machines.

use rand::Rng;
use sha2::{Digest, Sha256};

const ALPHABET: &[u8] = b"23456789ABCDEFGHJKLMNPQRSTUVWXYZ"; // no 0/O/1/I

/// A new random token like `7Q2K-9XMP-4HVD-J3FA` (~80 bits).
pub fn generate() -> String {
    let mut rng = rand::rng();
    let chars: Vec<char> = (0..16)
        .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())] as char)
        .collect();
    chars
        .chunks(4)
        .map(|c| c.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("-")
}

/// Normalize what a person typed (case, dashes, spaces, confusable letters).
pub fn normalize(t: &str) -> String {
    t.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| match c.to_ascii_uppercase() {
            'O' => '0',
            'I' | 'L' => '1',
            c => c,
        })
        .collect()
}

/// 32-byte pre-shared key for the Noise handshake.
pub fn psk(token: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"tendril-cluster-psk-v1:");
    h.update(normalize(token).as_bytes());
    h.finalize().into()
}

/// Persisted token for this machine's cluster (created on first use).
pub fn load_or_create() -> anyhow::Result<String> {
    let dir = dirs::config_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("tendril");
    let path = dir.join("cluster-token");
    if let Ok(t) = std::fs::read_to_string(&path) {
        let t = t.trim().to_string();
        if !t.is_empty() {
            return Ok(t);
        }
    }
    std::fs::create_dir_all(&dir)?;
    let t = generate();
    std::fs::write(&path, &t)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens() {
        let t = generate();
        assert_eq!(t.len(), 19);
        assert_eq!(psk(&t), psk(&t.to_lowercase().replace('-', " ")));
        assert_ne!(psk("AAAA"), psk("AAAB"));
    }
}

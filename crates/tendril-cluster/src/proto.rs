//! Messages between the server (pool) and the agents on each machine.
//!
//! Engine types (detections, roles) travel as JSON strings: they use serde
//! features the compact binary format can't decode.

use serde::{Deserialize, Serialize};
use tendril_core::NodeProfile;

pub const PROTOCOL: u32 = 5;

/// Another role of a deployment, as a starting role needs to reach it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Peer {
    pub role: String,
    /// The peer machine's tunnel endpoint (`ip:port`).
    pub tunnel: String,
    /// The peer machine's LAN address (for engines on a trusted network).
    pub host: String,
    /// The peer process's ports: "" is its main port, others by name.
    pub ports: Vec<(String, u16)>,
}

/// What a role on a machine is doing.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RoleState {
    Downloading {
        file: String,
        done: u64,
        total: u64,
    },
    Starting,
    /// The process runs and listens on these ports ("" = main).
    Started {
        ports: Vec<(String, u16)>,
    },
    /// Engine-specific preparation (pulling a model, placing it).
    Preparing {
        what: String,
        done: u64,
        total: u64,
    },
    Ready,
    Failed {
        error: String,
        log: Vec<String>,
    },
    Exited {
        code: Option<i32>,
        log: Vec<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Msg {
    // ---- control ---------------------------------------------------------
    Hello {
        protocol: u32,
        version: String,
        profile: NodeProfile,
        /// Port of this machine's tunnel listener.
        tunnel_port: u16,
        /// Random per-process id (survives reconnects).
        machine: u64,
        /// Runs inside the server process (the server's own compute).
        local: bool,
        /// `Vec<Detection>` as JSON.
        engines: String,
    },
    Welcome {
        key: String,
        cluster: String,
        /// The address the server sees for this machine.
        address: String,
    },
    Reject {
        reason: String,
    },
    Ping {
        t: u64,
    },
    Pong {
        t: u64,
    },
    /// Server → agent: look for engines again (after installing one).
    Detect,
    /// Agent → server: engines found (`Vec<Detection>` as JSON).
    Engines {
        engines: String,
    },
    /// Server → agent: start one role of a deployment.
    Start {
        deployment: String,
        /// `Role` as JSON.
        role: String,
        peers: Vec<Peer>,
        /// Isolation name for engines with their own discovery.
        namespace: String,
    },
    Stop {
        deployment: String,
        role: String,
    },
    /// Agent → server.
    RoleUpdate {
        deployment: String,
        role: String,
        state: RoleState,
    },
    /// End the session. `leave`: the agent process exits.
    Bye {
        reason: String,
        leave: bool,
    },

    // ---- tunnels -----------------------------------------------------------
    /// Open a byte stream to a role's main port on this machine
    /// (`deployment/role`). The answer is `Opened` or `Reject`; after
    /// `Opened` both sides send raw frames, an empty one ending each side.
    Open {
        target: String,
    },
    Opened,
}

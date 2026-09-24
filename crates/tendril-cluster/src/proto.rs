//! Messages exchanged between the coordinator and agents.

use serde::{Deserialize, Serialize};
use tendril_core::NodeProfile;
use tendril_engine::model::StageSpec;
use tendril_engine::sampler::SamplingParams;

pub const PROTOCOL: u32 = 1;

/// Activation tensors on the wire (raw little-endian bytes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireTensor {
    /// "f32", "bf16" or "f16".
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Payload {
    Tokens(Vec<u32>),
    Hidden(WireTensor),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleSetup {
    pub params: SamplingParams,
    /// Prompt tokens, for repetition penalties.
    pub history: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub len: u64,
}

/// Time one stage spent on one message, microseconds.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct StageTime {
    pub stage: u32,
    /// Waiting in the stage's queue.
    pub queue_us: u32,
    /// Running the stage (including (de)serializing activations).
    pub compute_us: u32,
}

/// Where a stage sends its output.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NextHop {
    /// Another stage's data port.
    Stage(String),
    /// The coordinator's data port (last stage).
    Coordinator(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Msg {
    // ---- control plane --------------------------------------------------
    Hello {
        protocol: u32,
        version: String,
        profile: NodeProfile,
        data_port: u16,
        backends: Vec<String>,
    },
    Welcome {
        node_id: u32,
        name: String,
        cluster: String,
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
    /// Bandwidth probe; the agent answers with ProbeAck.
    Probe {
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    ProbeAck {
        len: u64,
    },
    LoadStage {
        epoch: u64,
        index: u32,
        model_key: String,
        config_json: String,
        spec: StageSpec,
        format: String,
        device: String,
        tensors: Vec<TensorEntry>,
        next: NextHop,
    },
    /// Agent → coordinator: send me these tensors (not cached locally).
    NeedWeights {
        epoch: u64,
        names: Vec<String>,
    },
    WeightData {
        epoch: u64,
        name: String,
        offset: u64,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    WeightsDone {
        epoch: u64,
    },
    LoadProgress {
        epoch: u64,
        phase: String,
        done: u64,
        total: u64,
    },
    StageReady {
        epoch: u64,
        weight_bytes: u64,
        load_ms: u64,
        device: String,
    },
    LoadFailed {
        epoch: u64,
        error: String,
    },
    Unload {
        epoch: u64,
    },
    Bye {
        reason: String,
    },

    // ---- data plane -----------------------------------------------------
    /// Identify a data connection's sender.
    /// `results`: true when the sender is the last stage returning tokens to
    /// the coordinator; false when it feeds a stage's input.
    DataHello {
        epoch: u64,
        results: bool,
    },
    Forward {
        epoch: u64,
        seq: u64,
        pos: u32,
        payload: Payload,
        want_logits: bool,
        sample: Option<SampleSetup>,
        /// Per-stage timings appended as the message travels the pipeline.
        trace: Vec<StageTime>,
    },
    /// Sampled token from the last stage.
    Token {
        epoch: u64,
        seq: u64,
        pos: u32,
        token: u32,
        trace: Vec<StageTime>,
    },
    /// Free a sequence's KV on every stage (travels down the chain).
    Release {
        epoch: u64,
        seq: u64,
    },
    /// A stage failed on a sequence (travels down the chain to the coordinator).
    StageError {
        epoch: u64,
        seq: u64,
        stage: u32,
        error: String,
    },
}

impl WireTensor {
    pub fn from_tensor(t: &candle_core::Tensor) -> anyhow::Result<WireTensor> {
        use candle_core::DType;
        let shape = t.dims().to_vec();
        let flat = t.flatten_all()?;
        let (dtype, data) = match t.dtype() {
            DType::F32 => ("f32", bytes_of(&flat.to_vec1::<f32>()?)),
            DType::BF16 => (
                "bf16",
                bytes_of(
                    &flat
                        .to_vec1::<half::bf16>()?
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<u16>>(),
                ),
            ),
            DType::F16 => (
                "f16",
                bytes_of(
                    &flat
                        .to_vec1::<half::f16>()?
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<u16>>(),
                ),
            ),
            other => anyhow::bail!("unsupported activation dtype {other:?}"),
        };
        Ok(WireTensor {
            dtype: dtype.into(),
            shape,
            data,
        })
    }

    pub fn to_tensor(&self, device: &candle_core::Device) -> anyhow::Result<candle_core::Tensor> {
        use candle_core::Tensor;
        let n: usize = self.shape.iter().product();
        let elem = match self.dtype.as_str() {
            "f32" => 4,
            "bf16" | "f16" => 2,
            d => anyhow::bail!("unsupported wire dtype {d}"),
        };
        if self.shape.len() > 4 || n.checked_mul(elem) != Some(self.data.len()) {
            anyhow::bail!(
                "activation tensor has {} bytes for shape {:?} ({})",
                self.data.len(),
                self.shape,
                self.dtype
            );
        }
        let t = match self.dtype.as_str() {
            "f32" => {
                let v: Vec<f32> = self
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Tensor::from_vec(v, self.shape.clone(), device)?
            }
            "bf16" => {
                let v: Vec<half::bf16> = self
                    .data
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                Tensor::from_vec(v, self.shape.clone(), device)?
            }
            _ => {
                let v: Vec<half::f16> = self
                    .data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                Tensor::from_vec(v, self.shape.clone(), device)?
            }
        };
        Ok(t)
    }
}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    let mut out = vec![0u8; n];
    // SAFETY: plain-old-data copy; little-endian hosts (x86_64, aarch64).
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, out.as_mut_ptr(), n) };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_roundtrip() {
        let t = candle_core::Tensor::new(&[[1.5f32, -2.0], [3.25, 4.0]], &candle_core::Device::Cpu)
            .unwrap();
        let w = WireTensor::from_tensor(&t).unwrap();
        let back = w.to_tensor(&candle_core::Device::Cpu).unwrap();
        assert_eq!(back.to_vec2::<f32>().unwrap(), t.to_vec2::<f32>().unwrap());
        let mut bad = w.clone();
        bad.shape = vec![100, 100];
        assert!(bad.to_tensor(&candle_core::Device::Cpu).is_err());
    }
}

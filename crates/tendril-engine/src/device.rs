//! Device and precision selection.

use anyhow::Result;
use candle_core::{DType, Device};

/// Which accelerator this build can use, best first.
pub fn best_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return Ok(d);
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return Ok(d);
        }
    }
    Ok(Device::Cpu)
}

pub fn device_from_name(name: &str) -> Result<Device> {
    match name {
        "auto" => best_device(),
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        "metal" | "gpu" => Ok(Device::new_metal(0)?),
        #[cfg(feature = "cuda")]
        "cuda" | "gpu" => Ok(Device::new_cuda(0)?),
        other => anyhow::bail!(
            "device '{other}' is not available in this build (built with: {})",
            compiled_backends().join(", ")
        ),
    }
}

pub fn compiled_backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    if cfg!(feature = "metal") {
        v.push("metal");
    }
    if cfg!(feature = "cuda") {
        v.push("cuda");
    }
    v
}

pub fn device_label(d: &Device) -> &'static str {
    match d {
        Device::Cpu => "CPU",
        Device::Metal(_) => "Metal",
        Device::Cuda(_) => "CUDA",
    }
}

/// Activation dtype for a device and the checkpoint's stored dtype.
pub fn activation_dtype(device: &Device, stored: &str) -> DType {
    match device {
        // Our CPU kernels accumulate in f32; activations stay f32.
        Device::Cpu => DType::F32,
        _ => match stored {
            "float16" => DType::F16,
            "float32" => DType::F32,
            _ => DType::BF16,
        },
    }
}

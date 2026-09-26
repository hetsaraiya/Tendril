"""Generate tiny random-weight models for every supported architecture using
HuggingFace transformers, plus reference logits, so Tendril's engine can be
checked against the reference implementation without downloading anything.

    python tools/make_test_models.py OUT_DIR
"""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM
from transformers import (LlamaConfig, MistralConfig, Qwen2Config, Qwen3Config, GemmaConfig,
                          Gemma2Config, Gemma3TextConfig, Phi3Config)

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
V = 384
common = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=5,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
              rms_norm_eps=1e-6, bos_token_id=1, eos_token_id=2, pad_token_id=0)

cases = {
    "llama": LlamaConfig(**common, tie_word_embeddings=False, rope_theta=10000.0),
    "llama3": LlamaConfig(**common, tie_word_embeddings=True, rope_theta=500000.0,
                          rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                        "high_freq_factor": 4.0, "original_max_position_embeddings": 64}),
    "mistral": MistralConfig(**common, sliding_window=7),
    "qwen2": Qwen2Config(**common, tie_word_embeddings=True),
    "qwen3": Qwen3Config(**{**common, "head_dim": 32}, tie_word_embeddings=True),
    "gemma": GemmaConfig(**{**common, "head_dim": 16}, hidden_activation="gelu_pytorch_tanh"),
    "gemma2": Gemma2Config(**{**common, "head_dim": 16}, sliding_window=6, query_pre_attn_scalar=16,
                           attn_logit_softcapping=50.0, final_logit_softcapping=30.0),
    "gemma3": Gemma3TextConfig(**{**common, "head_dim": 16, "num_hidden_layers": 6}, sliding_window=6,
                               query_pre_attn_scalar=16, rope_local_base_freq=10000.0, rope_theta=1000000.0,
                               rope_scaling={"rope_type": "linear", "factor": 8.0}),
    "phi3": Phi3Config(**{**common, "num_key_value_heads": 4}, tie_word_embeddings=False),
}

ids = torch.tensor([[1] + [int(x) for x in torch.randint(3, V, (23,))]])
for name, cfg in cases.items():
    cfg._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").float().eval()
    # Random norms so the (1 + w) Gemma convention is actually exercised.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.normal_(0.0, 0.3)
                if "gemma" not in name:
                    p.add_(1.0)
    d = os.path.join(out, name)
    model.save_pretrained(d, safe_serialization=True)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    json.dump({"ids": ids[0].tolist(), "logits": logits.tolist()}, open(os.path.join(d, "reference.json"), "w"))
    # bf16 copy of weights for the native-precision path.
    model.to(torch.bfloat16).save_pretrained(d + "-bf16", safe_serialization=True)
    with torch.no_grad():
        l16 = model(ids).logits[0].float()
    json.dump({"ids": ids[0].tolist(), "logits": l16.tolist()}, open(os.path.join(d + "-bf16", "reference.json"), "w"))
    print(name, "ok", tuple(logits.shape))

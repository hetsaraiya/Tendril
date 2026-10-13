"""A tiny random llama GGUF (SPM vocab with byte fallback) for end-to-end tests.

    pip install gguf numpy
    python tools/make_tiny_gguf.py /tmp/tiny.gguf
"""
import sys
import numpy as np
import gguf
out = sys.argv[1]
L, H, FF, NH, NKV, V = 8, 128, 256, 4, 2, 300
w = gguf.GGUFWriter(out, "llama")
w.add_name("tendril-tiny")
w.add_context_length(2048)
w.add_embedding_length(H)
w.add_block_count(L)
w.add_feed_forward_length(FF)
w.add_head_count(NH)
w.add_head_count_kv(NKV)
w.add_rope_dimension_count(H // NH)
w.add_layer_norm_rms_eps(1e-5)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
toks, scores, types = ["<unk>", "<s>", "</s>"], [0.0, 0.0, 0.0], [2, 3, 3]
for b in range(256):
    toks.append(f"<0x{b:02X}>"); scores.append(0.0); types.append(6)
words = ["▁the", "▁a", "▁to", "▁is", "▁of", "▁and", "▁in", "▁it", "▁hello", "▁world", "▁cat", "▁dog", "▁run", "▁model", "▁split", "▁tendril", "e", "t", "a", "o", "i", "n", "s", "r", "h", "l", "d", "c", "u", "m", "▁", "!", ".", ",", "?", "▁you", "▁we", "▁yes", "▁no", "▁ok", "▁fast"]
for i, t in enumerate(words[: V - len(toks)]):
    toks.append(t); scores.append(-float(i)); types.append(1)
V = len(toks)
w.add_tokenizer_model("llama")
w.add_token_list(toks)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0)
w.add_add_bos_token(True)
w.add_chat_template("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:")
rng = np.random.default_rng(0)
def t(name, *shape):
    w.add_tensor(name, (rng.standard_normal(shape) * 0.05).astype(np.float32))
t("token_embd.weight", V, H)
for i in range(L):
    p = f"blk.{i}."
    w.add_tensor(p + "attn_norm.weight", np.ones(H, np.float32))
    t(p + "attn_q.weight", H, H)
    t(p + "attn_k.weight", H // NH * NKV, H)
    t(p + "attn_v.weight", H // NH * NKV, H)
    t(p + "attn_output.weight", H, H)
    w.add_tensor(p + "ffn_norm.weight", np.ones(H, np.float32))
    t(p + "ffn_gate.weight", FF, H)
    t(p + "ffn_up.weight", FF, H)
    t(p + "ffn_down.weight", H, FF)
w.add_tensor("output_norm.weight", np.ones(H, np.float32))
t("output.weight", V, H)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, V, "tokens")

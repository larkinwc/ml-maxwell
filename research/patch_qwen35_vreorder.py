#!/usr/bin/env python
"""Idempotent patch: invert llama.cpp's GDN V-head reorder for Qwen3.5 GGUF
when linear_num_value_heads != linear_num_key_heads (e.g. 9B: 32 vs 16).

The reorder is applied by wrapping the `weights` iterator at the top-level
Qwen3_5ForCausalLMBase.load_weights, BEFORE AutoWeightsLoader splits/strips
names. At this point GDN tensors still carry their full
``model.layers.N.linear_attn.X`` names.
"""
F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/models/qwen3_5.py"
s = open(F).read()

MARKER = "_qwen35_inv_v_head_reorder"
if MARKER in s:
    print("V-head reorder already patched; nothing to do.")
    raise SystemExit(0)

# --- 1. module-level helpers, inserted before the @support_torch_compile
#        decorator that precedes class Qwen3_5Model ---
helper = '''
def _qwen35_inv_v_head_reorder(t, dim, num_k, num_v_per_k, head_dim):
    """Invert llama.cpp's tiled GDN V-head layout back to HF grouped order.

    llama.cpp stores the GDN value-head axis as
    ``[num_v_per_k, num_k, head_dim]`` (tiled); HF / vLLM expect
    ``[num_k, num_v_per_k, head_dim]`` (grouped). This permutes the
    size-(num_v_per_k*num_k*head_dim) block along ``dim`` back to grouped.
    For ``dim`` == row dim this is a pure row permutation, lossless on
    packed-quantized GGUF byte tensors; out_proj needs a column permute and
    must be dequantized first.
    """
    shape = list(t.shape)
    d = dim % len(shape)
    new_shape = shape[:d] + [num_v_per_k, num_k, head_dim] + shape[d + 1:]
    perm = list(range(len(new_shape)))
    perm[d], perm[d + 1] = perm[d + 1], perm[d]
    return t.reshape(*new_shape).permute(*perm).contiguous().reshape(*shape)


def _qwen35_reorder_gdn_weights(weights, text_config):
    """Generator wrapping a GGUF weight stream, inverting llama.cpp's GDN
    V-head reorder for Qwen3.5 checkpoints whose value-head count differs
    from the key-head count. A no-op (pass-through) otherwise.
    """
    num_k = getattr(text_config, "linear_num_key_heads", 0) or 0
    num_v = getattr(text_config, "linear_num_value_heads", 0) or 0
    head_v_dim = getattr(text_config, "linear_value_head_dim", 0) or 0
    if not (num_k > 0 and num_v != num_k):
        yield from weights
        return
    num_v_per_k = num_v // num_k
    vdim = num_v * head_v_dim
    orig_qtype: dict[str, int] = {}
    for name, w in weights:
        base = name
        kind = None
        for suf in (".qweight_type", ".qweight", ".weight"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                kind = suf
                break
        if "linear_attn." not in base:
            yield name, w
            continue
        is_type = kind == ".qweight_type"
        if base.endswith("linear_attn.out_proj"):
            if is_type:
                orig_qtype[base] = int(w.item())
                # mark this layer's out_proj as unquantized fp16
                # (gguf WeightType.F16 == 1) so the runtime matmul takes the
                # x @ w.T branch on the reordered (dequantized) weight.
                yield name, torch.tensor(1, dtype=w.dtype)
                continue
            import gguf as _gguf
            qt = orig_qtype.get(base)
            if qt is not None and _gguf.GGMLQuantizationType(qt).name not in (
                "F32", "F16", "BF16"
            ):
                from gguf.quants import dequantize as _deq
                deq = _deq(w.cpu().numpy(), _gguf.GGMLQuantizationType(qt))
                ww = torch.from_numpy(deq.astype("float32"))
            else:
                ww = w.float()
            # ww is [out=hidden, in=value_dim]; reorder columns (dim 1)
            ww = _qwen35_inv_v_head_reorder(ww, 1, num_k, num_v_per_k, head_v_dim)
            yield name, ww.to(torch.float16)
            continue
        if is_type:
            yield name, w
            continue
        if base.endswith("linear_attn.in_proj_qkv") or base.endswith(
            "linear_attn.conv1d"
        ):
            head = w[:-vdim]
            tail = _qwen35_inv_v_head_reorder(
                w[-vdim:], 0, num_k, num_v_per_k, head_v_dim
            )
            yield name, torch.cat([head, tail], dim=0)
            continue
        if base.endswith("linear_attn.in_proj_z"):
            yield name, _qwen35_inv_v_head_reorder(
                w, 0, num_k, num_v_per_k, head_v_dim
            )
            continue
        if base.endswith("linear_attn.in_proj_b") or base.endswith(
            "linear_attn.in_proj_a"
        ):
            yield name, _qwen35_inv_v_head_reorder(w, 0, num_k, num_v_per_k, 1)
            continue
        if base.endswith("linear_attn.A_log") or base.endswith(
            "linear_attn.dt_bias"
        ):
            yield name, _qwen35_inv_v_head_reorder(w, 0, num_k, num_v_per_k, 1)
            continue
        yield name, w


'''
anchor1 = "@support_torch_compile(\n    dynamic_arg_dims={\n        \"input_ids\": 0,"
assert anchor1 in s, "could not find Qwen3_5Model decorator anchor"
s = s.replace(anchor1, helper + anchor1, 1)

# --- 2. wrap the iterator in Qwen3_5ForCausalLMBase.load_weights ---
old_lw = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)'''
new_lw = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Invert llama.cpp's GDN V-head reorder (no-op unless the GGUF's
        # value-head count differs from its key-head count, e.g. 9B).
        weights = _qwen35_reorder_gdn_weights(
            weights, self.config.get_text_config()
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)'''
assert old_lw in s, "could not find Base load_weights"
s = s.replace(old_lw, new_lw, 1)

open(F, "w").write(s)
print("Patched V-head reorder (iterator-wrap) into qwen3_5.py")

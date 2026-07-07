#!/usr/bin/env python3
"""Patch vLLM to support GGUF tuple shard-ids (Qwen3.5 GDN in_proj_qkv -> qkvz
shards (0,1,2)). GGUF stores q,k,v pre-fused as a single tensor 'attn_qkv', and
the gate 'attn_gate' as shard 3. The stock GGUF merged-linear loader rejects
tuple shard-ids; this patch makes the loader accept them and orders shards
canonically (q,k,v before z) in both the padded-weight builder and apply().
Idempotent."""
import os

ROOT = "/home/larkinwc/maxwell-stack/vllm-maxwell-core"
LIN = os.path.join(ROOT, "vllm/model_executor/layers/linear.py")
GGUF = os.path.join(ROOT, "vllm/model_executor/layers/quantization/gguf.py")

# ---------- linear.py: allow tuple shard-id for GGUF ----------
with open(LIN) as f:
    s = f.read()

old_reject = (
    '        if isinstance(loaded_shard_id, tuple) and (\n'
    '            is_gguf_weight or is_gguf_weight_type\n'
    '        ):\n'
    '            raise NotImplementedError(\n'
    '                "Shard id with multiple indices is not supported for GGUF."\n'
    '            )\n'
    '        if is_gguf_weight_type:\n'
    '            if loaded_shard_id is not None:\n'
    '                param.data[loaded_shard_id].copy_(loaded_weight)\n'
    '                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()\n'
    '            else:\n'
)
new_accept = (
    '        # GGUF: a tuple shard-id means the GGUF checkpoint already stores\n'
    '        # these consecutive shards pre-fused in a single tensor (e.g. Qwen3.5\n'
    '        # GDN in_proj_qkv -> qkvz shards (0,1,2)). Accept it as one shard.\n'
    '        if is_gguf_weight_type:\n'
    '            if isinstance(loaded_shard_id, tuple):\n'
    '                for _i in loaded_shard_id:\n'
    '                    param.data[_i].copy_(loaded_weight)\n'
    '                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()\n'
    '            elif loaded_shard_id is not None:\n'
    '                param.data[loaded_shard_id].copy_(loaded_weight)\n'
    '                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()\n'
    '            else:\n'
)
if "GGUF: a tuple shard-id means" not in s:
    assert old_reject in s, "linear.py reject block not found (maybe already patched/changed)"
    s = s.replace(old_reject, new_accept, 1)
    with open(LIN, "w") as f:
        f.write(s)
    print("linear.py: patched to accept GGUF tuple shard-id")
else:
    print("linear.py: already patched, skipping")

# ---------- gguf.py: canonical shard ordering ----------
with open(GGUF) as f:
    g = f.read()

if "_gguf_shard_sort_key" not in g:
    # Insert helper before class GGUFLinearMethod
    helper = (
        '_QKV_SHARD_ORDER = {"q": 0, "k": 1, "v": 2}\n'
        '\n'
        '\n'
        'def _gguf_shard_sort_key(idx):\n'
        '    """Canonical position of a GGUF merged-linear shard id.\n'
        '    Handles str q/k/v, int shard ids, and tuple (pre-fused) shard ids."""\n'
        '    if isinstance(idx, tuple):\n'
        '        return idx[0]\n'
        '    if isinstance(idx, str):\n'
        '        return _QKV_SHARD_ORDER.get(idx, 0)\n'
        '    return idx\n'
        '\n'
        '\n'
        'class GGUFLinearMethod(LinearMethodBase):\n'
    )
    anchor = 'class GGUFLinearMethod(LinearMethodBase):\n'
    assert anchor in g, "GGUFLinearMethod anchor not found"
    g = g.replace(anchor, helper, 1)
    print("gguf.py: inserted _gguf_shard_sort_key helper")
else:
    print("gguf.py: helper already present, skipping")

# Rewrite the padded-weight loop to use canonical sorted order.
old_loop = (
    '            # (dim0_start, dim0_end, dim1_size)\n'
    '            shard_offset_map = dict[str, tuple[int, int, int]]()\n'
    '            for idx in shard_id:\n'
    '                id_in_container = shard_id_map[idx]\n'
    '                start = sum(x.size(0) for x in data_container[:id_in_container])\n'
    '                end = start + data_container[id_in_container].size(0)\n'
    '                size = data_container[id_in_container].size(1)\n'
    '                padded_data[start:end, :size] = data_container[id_in_container]\n'
    '                shard_offset_map[idx] = (start, end, size)\n'
)
new_loop = (
    '            # (dim0_start, dim0_end, dim1_size). Place shards in canonical\n'
    '            # order (q,k,v,z) regardless of checkpoint load order so the\n'
    '            # fused layout is correct.\n'
    '            shard_offset_map = dict[str, tuple[int, int, int]]()\n'
    '            start = 0\n'
    '            for idx in sorted(shard_id, key=_gguf_shard_sort_key):\n'
    '                id_in_container = shard_id_map[idx]\n'
    '                data = data_container[id_in_container]\n'
    '                end = start + data.size(0)\n'
    '                size = data.size(1)\n'
    '                padded_data[start:end, :size] = data\n'
    '                shard_offset_map[idx] = (start, end, size)\n'
    '                start = end\n'
)
if "regardless of checkpoint load order" not in g:
    assert old_loop in g, "padded-weight loop not found"
    g = g.replace(old_loop, new_loop, 1)
    print("gguf.py: patched padded-weight loop ordering")
else:
    print("gguf.py: padded-weight loop already patched, skipping")

# Rewrite apply() shard ordering.
old_apply = (
    '        if shard_id:\n'
    '            # dequantize shard weights respectively\n'
    '            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id\n'
)
new_apply = (
    '        if shard_id:\n'
    '            # dequantize shard weights respectively, in canonical order\n'
    '            shard_id = (\n'
    '                ["q", "k", "v"]\n'
    '                if "q" in shard_id\n'
    '                else sorted(shard_id, key=_gguf_shard_sort_key)\n'
    '            )\n'
)
if "in canonical order" not in g:
    assert old_apply in g, "apply shard_id line not found"
    g = g.replace(old_apply, new_apply, 1)
    print("gguf.py: patched apply() ordering")
else:
    print("gguf.py: apply() already patched, skipping")

with open(GGUF, "w") as f:
    f.write(g)
print("gguf.py written")

# ---------- qwen3_5.py: reshape GGUF conv1d [C,K] -> [C,1,K] ----------
Q35 = os.path.join(ROOT, "vllm/model_executor/models/qwen3_5.py")
with open(Q35) as f:
    q = f.read()

old_call = (
    '                    weight_loader = getattr(\n'
    '                        param, "weight_loader", default_weight_loader\n'
    '                    )\n'
    '                    weight_loader(param, loaded_weight)\n'
)
new_call = (
    '                    weight_loader = getattr(\n'
    '                        param, "weight_loader", default_weight_loader\n'
    '                    )\n'
    '                    # GGUF stores conv1d weight as [channels, kernel] but the\n'
    '                    # Conv1d param is [channels, 1, kernel]; add the middle dim.\n'
    '                    if (\n'
    '                        "conv1d" in name\n'
    '                        and loaded_weight.dim() == 2\n'
    '                        and getattr(param, "dim", lambda: None)\n'
    '                        and param.dim() == 3\n'
    '                        and param.shape[1] == 1\n'
    '                    ):\n'
    '                        loaded_weight = loaded_weight.unsqueeze(1)\n'
    '                    # GGUF stores ssm_a as A = -exp(A_log) (already\n'
    '                    # exponentiated by llama.cpp), but vLLM\'s GDN kernel\n'
    '                    # expects raw A_log and applies -exp() internally.\n'
    '                    # Invert: A_log = log(-A).\n'
    '                    if name.endswith("linear_attn.A_log"):\n'
    '                        loaded_weight = torch.log(-loaded_weight.float())\n'
    '                    weight_loader(param, loaded_weight)\n'
)
if "GGUF stores conv1d weight as" not in q:
    assert old_call in q, "qwen3_5 else-branch weight_loader call not found"
    q = q.replace(old_call, new_call, 1)
    with open(Q35, "w") as f:
        f.write(q)
    print("qwen3_5.py: patched conv1d reshape")
else:
    print("qwen3_5.py: conv1d reshape already present, skipping")
print("DONE")

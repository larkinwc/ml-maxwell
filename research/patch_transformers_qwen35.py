#!/usr/bin/env python3
"""Patch vendored transformers to support Qwen3.5 GGUF config parsing.
Idempotent: checks for marker before editing."""
import re, sys, os

TR = "/home/larkinwc/miniconda3/envs/vllm-maxwell/lib/python3.12/site-packages/transformers"
GGML = os.path.join(TR, "integrations", "ggml.py")
MGUF = os.path.join(TR, "modeling_gguf_pytorch_utils.py")

# ---------- 1. ggml.py : add qwen35 to GGUF_CONFIG_MAPPING ----------
with open(GGML) as f:
    g = f.read()

if '"qwen3_5": {' not in g:
    # Insert a qwen3_5 mapping right after the "qwen3_moe" config block.
    # NOTE: key must be "qwen3_5" because the config loop replaces the GGUF
    # architecture token "qwen35" -> "qwen3_5" (updated_architecture) before
    # matching the prefix against this mapping.
    anchor = '    "falcon": {\n        "context_length": "max_position_embeddings",'
    qwen35_block = (
        '    "qwen3_5": {\n'
        '        "context_length": "max_position_embeddings",\n'
        '        "block_count": "num_hidden_layers",\n'
        '        "feed_forward_length": "intermediate_size",\n'
        '        "embedding_length": "hidden_size",\n'
        '        "rope.freq_base": "rope_theta",\n'
        '        "attention.head_count": "num_attention_heads",\n'
        '        "attention.head_count_kv": "num_key_value_heads",\n'
        '        "attention.key_length": "head_dim",\n'
        '        "attention.layer_norm_rms_epsilon": "rms_norm_eps",\n'
        '        "full_attention_interval": "full_attention_interval",\n'
        '        "ssm.conv_kernel": "linear_conv_kernel_dim",\n'
        '        "ssm.state_size": "linear_key_head_dim",\n'
        '        "ssm.group_count": "linear_num_key_heads",\n'
        '    },\n'
    )
    assert anchor in g, "falcon anchor not found in ggml.py"
    g = g.replace(anchor, qwen35_block + anchor, 1)
    with open(GGML, "w") as f:
        f.write(g)
    print("ggml.py: inserted qwen35 GGUF_CONFIG_MAPPING block")
else:
    print("ggml.py: qwen35 already present, skipping")

# ---------- 2. modeling_gguf_pytorch_utils.py ----------
with open(MGUF) as f:
    m = f.read()

# 2a. arch route: map general.architecture 'qwen35' -> model_type 'qwen3_5'
if 'updated_architecture = "qwen3_5"' not in m:
    route_anchor = '    elif "minimax-m2" in architecture:\n        updated_architecture = "minimax_m2"\n'
    route_add = (
        route_anchor
        + '    elif "qwen35" in architecture or "qwen3_5" in architecture:\n'
          '        updated_architecture = "qwen3_5"\n'
    )
    assert route_anchor in m, "minimax route anchor not found"
    m = m.replace(route_anchor, route_add, 1)
    print("modeling_gguf: inserted qwen35 arch route")
else:
    print("modeling_gguf: qwen35 arch route already present, skipping")

# 2a2. text-only GGUF: resolve qwen3_5 -> qwen3_5_text (no vision_config),
#      mirroring the gemma3 -> gemma3_text pattern.
if 'parsed_parameters["config"]["model_type"] == "qwen3_5"' not in m:
    g3_anchor = (
        '    if parsed_parameters["config"]["model_type"] == "gemma3":\n'
        '        parsed_parameters["config"]["model_type"] = "gemma3_text"\n'
    )
    g3_add = g3_anchor + (
        '\n    if parsed_parameters["config"]["model_type"] == "qwen3_5":\n'
        '        parsed_parameters["config"]["model_type"] = "qwen3_5_text"\n'
    )
    assert g3_anchor in m, "gemma3_text anchor not found"
    m = m.replace(g3_anchor, g3_add, 1)
    print("modeling_gguf: inserted qwen3_5 -> qwen3_5_text conversion")
else:
    print("modeling_gguf: qwen3_5_text conversion already present, skipping")

# 2b. post-process derived config fields (anchor before gpt_oss block)
if "QWEN35_DERIVED_FIELDS" not in m:
    pp_anchor = '    if updated_architecture == "gpt_oss":\n'
    pp_block = (
        '    # QWEN35_DERIVED_FIELDS: derive GDN/SSM + rope fields not directly in metadata\n'
        '    if updated_architecture == "qwen3_5":\n'
        '        cfg = parsed_parameters["config"]\n'
        '        ssm_state = read_field(reader, "qwen35.ssm.state_size")\n'
        '        ssm_inner = read_field(reader, "qwen35.ssm.inner_size")\n'
        '        ssm_group = read_field(reader, "qwen35.ssm.group_count")\n'
        '        rope_dim = read_field(reader, "qwen35.rope.dimension_count")\n'
        '        head_dim = cfg.get("head_dim")\n'
        '        if ssm_state:\n'
        '            cfg["linear_key_head_dim"] = ssm_state[0]\n'
        '            cfg["linear_value_head_dim"] = ssm_state[0]\n'
        '        if ssm_group:\n'
        '            cfg["linear_num_key_heads"] = ssm_group[0]\n'
        '        # value heads derive from inner_size / state_size (may differ from key heads)\n'
        '        if ssm_inner and ssm_state:\n'
        '            cfg["linear_num_value_heads"] = int(ssm_inner[0] // ssm_state[0])\n'
        '        elif ssm_group:\n'
        '            cfg["linear_num_value_heads"] = ssm_group[0]\n'
        '        if rope_dim and head_dim:\n'
        '            cfg["partial_rotary_factor"] = float(rope_dim[0]) / float(head_dim)\n'
        '        cfg.setdefault("mtp_num_hidden_layers", 1)\n'
        '        cfg.setdefault("hidden_act", "silu")\n'
        '\n'
    )
    assert pp_anchor in m, "gpt_oss anchor not found"
    m = m.replace(pp_anchor, pp_block + pp_anchor, 1)
    print("modeling_gguf: inserted qwen35 derived-fields block")
else:
    print("modeling_gguf: qwen35 derived fields already present, skipping")

with open(MGUF, "w") as f:
    f.write(m)
print("DONE")

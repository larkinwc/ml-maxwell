#!/usr/bin/env python3
"""Patch vLLM gguf_loader.py to support Qwen3.5 GGUF tensor mapping.
Adds manual overrides for the 2 GDN params gguf-py can't map:
  blk.N.ssm_a       -> model.layers.N.linear_attn.A_log   (no .weight suffix)
  blk.N.ssm_dt.bias -> model.layers.N.linear_attn.dt_bias (no suffix)
And maps model_type qwen3_5/qwen3_5_text -> gguf arch 'qwen35'.
Idempotent."""
import os

F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/model_loader/gguf_loader.py"
with open(F) as fh:
    s = fh.read()

if "QWEN35_OVERRIDES" not in s:
    anchor = "        arch = None\n        for key, value in gguf.MODEL_ARCH_NAMES.items():"
    block = (
        '        # QWEN35_OVERRIDES: Qwen3.5 hybrid (GatedDeltaNet SSM + full attn).\n'
        '        # gguf-py maps most tensors via the qwen35 arch map, but two GDN\n'
        '        # params are nn.Parameters without a .weight/.bias suffix and are\n'
        '        # not covered by the tensor-name map, so add them manually.\n'
        '        if model_type in ("qwen3_5", "qwen3_5_text"):\n'
        '            model_type = "qwen35"\n'
        '            for idx in range(text_config.num_hidden_layers):\n'
        '                gguf_to_hf_name_map[f"blk.{idx}.ssm_a"] = (\n'
        '                    f"model.layers.{idx}.linear_attn.A_log"\n'
        '                )\n'
        '                gguf_to_hf_name_map[f"blk.{idx}.ssm_dt.bias"] = (\n'
        '                    f"model.layers.{idx}.linear_attn.dt_bias"\n'
        '                )\n'
        '\n'
    )
    assert anchor in s, "arch=None anchor not found"
    s = s.replace(anchor, block + anchor, 1)
    with open(F, "w") as fh:
        fh.write(s)
    print("gguf_loader.py: inserted qwen35 overrides")
else:
    print("gguf_loader.py: qwen35 overrides already present, skipping")

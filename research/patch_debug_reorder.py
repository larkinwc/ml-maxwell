#!/usr/bin/env python
"""Temporary debug: log each GDN tensor the reorder touches (layer 0 only)."""
F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/models/qwen3_5.py"
s = open(F).read()

MARK = "# __DBG_REORDER__"
if MARK in s:
    print("debug already present")
    raise SystemExit(0)

old = '''        is_type = kind == ".qweight_type"
        if base.endswith("linear_attn.out_proj"):'''
new = '''        is_type = kind == ".qweight_type"
        if ".layers.0." in name:  # __DBG_REORDER__
            import sys as _sys
            _sys.stderr.write(
                f"[DBG-REORDER] name={name} kind={kind} "
                f"shape={tuple(loaded_weight.shape)} "
                f"dtype={loaded_weight.dtype}\\n"
            )
            _sys.stderr.flush()
        if base.endswith("linear_attn.out_proj"):'''
assert old in s
s = s.replace(old, new, 1)
open(F, "w").write(s)
print("debug print inserted")

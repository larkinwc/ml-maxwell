#!/usr/bin/env python
"""Persist out_proj qtype map on self (AutoWeightsLoader splits the GGUF
types-first/data-second ordering across multiple load_weights calls)."""
F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/models/qwen3_5.py"
s = open(F).read()

old = '''        _gdn_vdim = _gdn_num_v * _gdn_hv
        _gdn_orig_qtype: dict[str, int] = {}
        for name, loaded_weight in weights:'''
new = '''        _gdn_vdim = _gdn_num_v * _gdn_hv
        # Persisted on self: AutoWeightsLoader splits the GGUF iterator's
        # "all qweight_type first, then all qweight" ordering across multiple
        # load_weights() calls, so out_proj's qtype (seen in an earlier call)
        # must survive into the call that carries its packed qweight bytes.
        if not hasattr(self, "_gdn_orig_qtype"):
            self._gdn_orig_qtype = {}
        _gdn_orig_qtype = self._gdn_orig_qtype
        for name, loaded_weight in weights:'''

if "self._gdn_orig_qtype" in s:
    print("qtype persistence already applied.")
elif old in s:
    s = s.replace(old, new, 1)
    open(F, "w").write(s)
    print("Applied qtype persistence.")
else:
    raise SystemExit("anchor not found")

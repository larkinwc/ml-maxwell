#!/usr/bin/env python
F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/models/qwen3_5.py"
s = open(F).read()

anchor = '''        _gdn_orig_qtype = self._gdn_orig_qtype
        for name, loaded_weight in weights:
            if _gdn_need_reorder and "linear_attn." in name:'''
new = '''        _gdn_orig_qtype = self._gdn_orig_qtype
        _dbg_names = []
        for name, loaded_weight in weights:
            if ".layers.0." in name:
                _dbg_names.append(name)
            if _gdn_need_reorder and "linear_attn." in name:'''
assert anchor in s, "anchor not found"
s = s.replace(anchor, new, 1)

# log the collected names at end of loop (before "return loaded_params")
ret = '''            loaded_params.add(name)
        return loaded_params'''
new_ret = '''            loaded_params.add(name)
        logger.warning("[DBG-NAMES] layer0 names seen: %s", _dbg_names)
        return loaded_params'''
assert ret in s, "return anchor not found"
s = s.replace(ret, new_ret, 1)
open(F, "w").write(s)
print("added name logging")

#!/usr/bin/env python
F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/models/qwen3_5.py"
s = open(F).read()

# replace stderr debug with logger debug
old_dbg = '''        if ".layers.0." in name:  # __DBG_REORDER__
            import sys as _sys
            _sys.stderr.write(
                f"[DBG-REORDER] name={name} kind={kind} "
                f"shape={tuple(loaded_weight.shape)} "
                f"dtype={loaded_weight.dtype}\\n"
            )
            _sys.stderr.flush()
'''
new_dbg = '''        if ".layers.0." in name:  # __DBG_REORDER__
            logger.warning(
                "[DBG-REORDER] name=%s kind=%s shape=%s dtype=%s",
                name, kind, tuple(loaded_weight.shape), loaded_weight.dtype,
            )
'''
assert old_dbg in s, "old debug not found"
s = s.replace(old_dbg, new_dbg, 1)

# add precompute logging of gate decision
anchor = "        _gdn_need_reorder = _gdn_num_k > 0 and _gdn_num_v != _gdn_num_k\n"
add = anchor + ('        logger.warning(\n'
                '            "[DBG-GATE] num_k=%s num_v=%s hv=%s need_reorder=%s",\n'
                '            _gdn_num_k, _gdn_num_v, _gdn_hv, _gdn_need_reorder,\n'
                '        )\n')
assert anchor in s
s = s.replace(anchor, add, 1)

open(F, "w").write(s)
print("debug switched to logger + gate logging")

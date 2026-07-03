#!/usr/bin/env python
F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/models/qwen3_5.py"
s = open(F).read()

# change layer0-only collection to count ALL names + sample
old = '''        _dbg_names = []
        for name, loaded_weight in weights:
            if ".layers.0." in name:
                _dbg_names.append(name)'''
new = '''        _dbg_names = []
        _dbg_all = 0
        for name, loaded_weight in weights:
            _dbg_all += 1
            if _dbg_all <= 6 or ".layers.0." in name:
                _dbg_names.append(name)'''
assert old in s, "old block not found"
s = s.replace(old, new, 1)

old2 = '''        logger.warning("[DBG-NAMES] layer0 names seen: %s", _dbg_names)'''
new2 = '''        logger.warning("[DBG-NAMES] total=%s sample=%s", _dbg_all, _dbg_names[:12])'''
assert old2 in s
s = s.replace(old2, new2, 1)
open(F, "w").write(s)
print("patched debug4")

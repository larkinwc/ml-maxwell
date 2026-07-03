#!/usr/bin/env python3
"""Idempotent patch: fix per-rank TP narrowing of pre-fused GGUF qkv tuple shards.

The MergedColumnParallelLinear GGUF weight_loader narrowed a pre-fused tensor
(covering several consecutive output shards, e.g. Qwen3.5 GDN qkv -> qkvz shards
(0,1,2)) *contiguously* across TP ranks. Because the sub-shards have unequal
sizes (key_dim, key_dim, value_dim), that slicing scrambles q/k/v across ranks
at TP>=2 (rank0 got all-q, rank1 all-k, ...). Narrow each sub-shard
independently instead.
"""
import re, sys

F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/layers/linear.py"
MARKER = "# GGUF-TUPLE-TP-FIX"

src = open(F).read()
if MARKER in src:
    print("already patched")
    sys.exit(0)

old = """        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return"""

new = """        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)

            if loaded_shard_id is not None:
                if isinstance(loaded_shard_id, tuple):
                    # GGUF-TUPLE-TP-FIX: a pre-fused GGUF tensor covering
                    # several consecutive output shards (e.g. Qwen3.5 GDN qkv
                    # -> qkvz shards (0,1,2)). The sub-shards have unequal sizes
                    # so narrow each sub-block independently for TP instead of
                    # slicing the concatenated tensor contiguously (which would
                    # scramble q/k/v across ranks at TP>=2).
                    parts = []
                    off = 0
                    for _sid in loaded_shard_id:
                        _sz = self.output_sizes[_sid]
                        _full = loaded_weight.narrow(output_dim, off, _sz)
                        _ss = _sz // self.tp_size
                        parts.append(
                            _full.narrow(output_dim, self.tp_rank * _ss, _ss)
                        )
                        off += _sz
                    loaded_weight = torch.cat(parts, dim=output_dim)
                else:
                    shard_size = loaded_weight.size(output_dim) // self.tp_size
                    start_idx = self.tp_rank * shard_size
                    loaded_weight = loaded_weight.narrow(
                        output_dim, start_idx, shard_size
                    )
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return"""

if old not in src:
    print("ERROR: anchor text not found; aborting", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
open(F, "w").write(src)
print("patched OK")

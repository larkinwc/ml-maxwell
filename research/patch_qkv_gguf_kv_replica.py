#!/usr/bin/env python3
"""Idempotent patch: replicate KV heads in the QKVParallelLinear GGUF loader.

The GGUF weight_loader path in QKVParallelLinear sharded the q/k/v tensors by
`size // tp_size` with no regard for `num_kv_head_replicas`. When
`tp_size > total_num_kv_heads` (e.g. Qwen3.5-9B: 4 KV heads at TP=8), vLLM
replicates KV heads across ranks (num_kv_heads=1, num_kv_head_replicas=2) and
the downstream split expects each rank to hold one *full* KV head. The naive
GGUF narrow instead handed each rank only a fraction of a KV head, so the
qkv.split([...]) in qwen3_next.py forward crashed
(split_sizes=[1024,256,256] vs tensor 1280 at TP=8).

This mirrors what the non-GGUF path does via load_qkv_weight(num_heads=
num_kv_head_replicas): size k/v by KV-head count and offset by
tp_rank // num_kv_head_replicas so replicas read the same head.
"""
import sys

F = "/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/layers/linear.py"
MARKER = "# GGUF-QKV-KV-REPLICA-FIX"

src = open(F).read()
if MARKER in src:
    print("already patched")
    sys.exit(0)

# The QKV branch is uniquely identified by the trailing data_container append
# block (the MergedColumn variant was already rewritten and no longer matches).
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
                # GGUF-QKV-KV-REPLICA-FIX: replicate KV heads when
                # tp_size > total_num_kv_heads. The query shard divides
                # evenly across ranks, but each k/v shard must hand every
                # rank a whole number of KV heads, with num_kv_head_replicas
                # consecutive ranks sharing the same head(s) -- matching the
                # non-GGUF load_qkv_weight path. The previous code sliced by
                # size // tp_size for all of q/k/v, which split a single KV
                # head across ranks and broke the qkv split at TP>num_kv.
                if loaded_shard_id == "q":
                    shard_size = loaded_weight.size(output_dim) // self.tp_size
                    start_idx = self.tp_rank * shard_size
                else:
                    shard_size = (
                        loaded_weight.size(output_dim)
                        // self.total_num_kv_heads
                        * self.num_kv_heads
                    )
                    start_idx = (
                        self.tp_rank // self.num_kv_head_replicas
                    ) * shard_size
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return"""

cnt = src.count(old)
if cnt != 1:
    print(f"ERROR: expected exactly 1 match for QKV block, found {cnt}", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
open(F, "w").write(src)
print("patched OK")

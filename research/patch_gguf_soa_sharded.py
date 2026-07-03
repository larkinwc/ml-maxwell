#!/usr/bin/env python3
"""Idempotent: extend SoA Q4_0 to sharded/fused layers (qkv, gate_up).

Builds per-shard SoA buffers from the padded weight and routes each shard in
apply()'s sharded branch through the SoA op when the shard is Q4_0.
"""
F="/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/layers/quantization/gguf.py"
s=open(F).read()
if "qweight_soa_shards" in s:
    print("already"); raise SystemExit

# 1. Build per-shard SoA at the end of _create_padded_weight_param (after register_parameter)
build_anchor='''            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)'''
build_new='''            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)
            # Maxwell: build per-shard SoA Q4_0 buffers (qkv / gate_up).
            soa_shards = {}
            for idx, (start, end, size) in shard_offset_map.items():
                stype = layer.qweight_type.shard_weight_type.get(idx)
                if stype is None or int(stype) != int(WeightType.Q4_0):
                    continue
                sw = padded_data[start:end, :size].contiguous()
                srows = sw.shape[0]
                sbpr = sw.shape[1] // 18
                swb = sw.view(srows, sbpr, 18)
                sq = swb[:, :, 2:18].contiguous().view(srows, sbpr * 16)
                ss = swb[:, :, 0:2].contiguous().view(srows, sbpr * 2).view(
                    torch.float16).contiguous()
                soa_shards[idx] = (sq, ss)
            if soa_shards:
                layer.qweight_soa_shards = soa_shards'''
assert s.count(build_anchor)==1, f"build anchor={s.count(build_anchor)}"
s=s.replace(build_anchor, build_new, 1)

# 2. Route per-shard in apply()
apply_anchor='''            qweight = layer.qweight
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(
                    fused_mul_mat_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)'''
apply_new='''            qweight = layer.qweight
            soa_shards = getattr(layer, "qweight_soa_shards", None)
            disable_soa = os.environ.get("VLLM_DISABLE_Q4_0_SOA", "0") == "1"
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                if (not disable_soa and soa_shards is not None
                        and idx in soa_shards):
                    sq, ss = soa_shards[idx]
                    result.append(
                        fused_mul_mat_gguf_soa(
                            x, sq, ss,
                            qweight[start:end, :offset].contiguous(),
                            qweight_type, sq.shape[0],
                        )
                    )
                else:
                    result.append(
                        fused_mul_mat_gguf(
                            x, qweight[start:end, :offset].contiguous(),
                            qweight_type,
                        )
                    )
            out = torch.cat(result, axis=1)'''
assert s.count(apply_anchor)==1, f"apply anchor={s.count(apply_anchor)}"
s=s.replace(apply_anchor, apply_new, 1)

# 3. ensure os imported
if "\nimport os\n" not in s and not s.startswith("import os"):
    s=s.replace("import gguf\n","import os\nimport gguf\n",1)

open(F,"w").write(s)
print("patched sharded SoA")

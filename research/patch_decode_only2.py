#!/usr/bin/env python3
"""Part 2: route apply() correctly when AoS qweight is freed (decode-only)."""
F="/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/layers/quantization/gguf.py"
s=open(F).read()
if "qweight_soa_row" not in s:
    print("ERROR: run part1 first"); raise SystemExit(1)
if "force_soa=_gguf_decode_only()" in s:
    print("already part2"); raise SystemExit

# A. sharded build: free padded qweight in decode-only mode
old_shard_end='''                soa_shards[idx] = (sq, ss)
            if soa_shards:
                layer.qweight_soa_shards = soa_shards'''
new_shard_end='''                soa_shards[idx] = (sq, ss)
            if soa_shards:
                layer.qweight_soa_shards = soa_shards
                if _gguf_decode_only() and len(soa_shards) == len(
                        shard_offset_map):
                    # all shards are Q4_0 SoA -> drop the padded AoS copy
                    _free_param_data(layer.qweight)'''
assert s.count(old_shard_end)==1, f"shard_end={s.count(old_shard_end)}"
s=s.replace(old_shard_end, new_shard_end, 1)

# B. sharded apply: when force_soa, don't slice the (possibly freed) qweight
old_apply_shard='''            qweight = layer.qweight
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
new_apply_shard='''            qweight = layer.qweight
            soa_shards = getattr(layer, "qweight_soa_shards", None)
            decode_only = _gguf_decode_only()
            disable_soa = (not decode_only) and (
                os.environ.get("VLLM_DISABLE_Q4_0_SOA", "0") == "1")
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                if (not disable_soa and soa_shards is not None
                        and idx in soa_shards):
                    sq, ss = soa_shards[idx]
                    # In decode-only mode the AoS copy is freed; pass sq as an
                    # unused placeholder for the (skipped) AoS fallback arg.
                    aos = sq if decode_only else qweight[
                        start:end, :offset].contiguous()
                    result.append(
                        fused_mul_mat_gguf_soa(
                            x, sq, ss, aos, qweight_type, sq.shape[0],
                            decode_only,
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
assert s.count(old_apply_shard)==1, f"apply_shard={s.count(old_apply_shard)}"
s=s.replace(old_apply_shard, new_apply_shard, 1)

# C. non-sharded apply: use stored row, pass force_soa, placeholder for freed AoS
old_apply_ns='''        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            if _has_q4_0_soa(layer):
                out = fused_mul_mat_gguf_soa(
                    x,
                    layer.qweight_soa_quants,
                    layer.qweight_soa_scales,
                    qweight,
                    qweight_type,
                    qweight.shape[0],
                )
            else:
                out = fused_mul_mat_gguf(x, qweight, qweight_type)'''
new_apply_ns='''        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            if _has_q4_0_soa(layer):
                decode_only = _gguf_decode_only()
                row = getattr(layer, "qweight_soa_row", None)
                if row is None:
                    row = qweight.shape[0]
                aos = layer.qweight_soa_quants if decode_only else qweight
                out = fused_mul_mat_gguf_soa(
                    x,
                    layer.qweight_soa_quants,
                    layer.qweight_soa_scales,
                    aos,
                    qweight_type,
                    row,
                    decode_only,
                )
            else:
                out = fused_mul_mat_gguf(x, qweight, qweight_type)'''
assert s.count(old_apply_ns)==1, f"apply_ns={s.count(old_apply_ns)}"
s=s.replace(old_apply_ns, new_apply_ns, 1)

open(F,"w").write(s)
print("patched part2 (apply routing + sharded free)")

#!/usr/bin/env python3
"""Make VLLM_DISABLE_Q4_0_SOA=1 also skip BUILDING the SoA buffers (true AoS
baseline, no extra memory). And the sharded build path likewise."""
F="/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/layers/quantization/gguf.py"
s=open(F).read()
if "def _soa_disabled()" in s:
    print("already"); raise SystemExit

# add a helper next to _gguf_decode_only
anchor='''def _gguf_decode_only() -> bool:
    return os.environ.get("VLLM_GGUF_DECODE_ONLY", "0") == "1"'''
new=anchor+'''


def _soa_disabled() -> bool:
    # decode-only implies SoA is required; otherwise honor the disable flag.
    if _gguf_decode_only():
        return False
    return os.environ.get("VLLM_DISABLE_Q4_0_SOA", "0") == "1"'''
assert s.count(anchor)==1, f"anchor={s.count(anchor)}"
s=s.replace(anchor,new,1)

# non-sharded build: bail early when disabled
nb_anchor='''def _maybe_build_q4_0_soa(layer: torch.nn.Module) -> None:
    """Build SoA quants/scales for a non-sharded Q4_0 qweight, stash on layer."""
    qweight = layer.qweight'''
nb_new='''def _maybe_build_q4_0_soa(layer: torch.nn.Module) -> None:
    """Build SoA quants/scales for a non-sharded Q4_0 qweight, stash on layer."""
    if _soa_disabled():
        return
    qweight = layer.qweight'''
assert s.count(nb_anchor)==1, f"nb_anchor={s.count(nb_anchor)}"
s=s.replace(nb_anchor,nb_new,1)

# sharded build: guard the soa_shards loop
sb_anchor='''            # Maxwell: build per-shard SoA Q4_0 buffers (qkv / gate_up).
            soa_shards = {}'''
sb_new='''            # Maxwell: build per-shard SoA Q4_0 buffers (qkv / gate_up).
            soa_shards = {}
            if not _soa_disabled():'''
assert s.count(sb_anchor)==1, f"sb_anchor={s.count(sb_anchor)}"
s=s.replace(sb_anchor,sb_new,1)
# now indent the existing for-loop body that builds soa_shards by 4 spaces
old_loop='''            if not _soa_disabled():
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
                soa_shards[idx] = (sq, ss)'''
new_loop='''            if not _soa_disabled():
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
                    soa_shards[idx] = (sq, ss)'''
assert s.count(old_loop)==1, f"old_loop={s.count(old_loop)}"
s=s.replace(old_loop,new_loop,1)

open(F,"w").write(s)
print("patched skip-build")

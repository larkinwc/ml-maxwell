#!/usr/bin/env python3
"""Idempotent: add VLLM_GGUF_DECODE_ONLY mode that frees the AoS Q4_0 copy.

When VLLM_GGUF_DECODE_ONLY=1, after building the SoA quants/scales buffers we
drop the original 18-byte AoS qweight (saving ~3.5 GB model-wide for 9B) and
route ALL batch sizes through the SoA op (which is correct for batch>1, just
slower than MMQ for big prefill batches -- acceptable on a decode-only tier).
Default (flag unset) is unchanged: AoS kept, MMQ used for prefill.
"""
F="/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/layers/quantization/gguf.py"
s=open(F).read()
if "_GGUF_DECODE_ONLY" in s:
    print("already"); raise SystemExit

# 1. helper to read the env flag (module level, after _has_q4_0_soa)
anchor_has='''def _has_q4_0_soa(layer: torch.nn.Module) -> bool:
    if os.environ.get("VLLM_DISABLE_Q4_0_SOA", "0") == "1":
        return False
    return getattr(layer, "qweight_soa_quants", None) is not None'''
new_has=anchor_has+'''


def _gguf_decode_only() -> bool:
    return os.environ.get("VLLM_GGUF_DECODE_ONLY", "0") == "1"


def _free_param_data(param: torch.nn.Parameter) -> None:
    """Shrink a parameter's storage to ~nothing while keeping its attrs."""
    param.data = torch.empty(0, dtype=param.data.dtype, device=param.data.device)'''
assert s.count(anchor_has)==1, f"anchor_has={s.count(anchor_has)}"
s=s.replace(anchor_has, new_has, 1)

# 2. _fused_mul_mat_gguf_soa: add force_soa to skip AoS fallback
old_soa='''def _fused_mul_mat_gguf_soa(
    x: torch.Tensor,
    quants: torch.Tensor,
    scales: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    row: int,
) -> torch.Tensor:
    # Maxwell SoA Q4_0 path: 128-bit coalesced quant loads for the decode
    # (vec) kernel. Prefill / large batch falls back to the AoS MMQ kernel.
    mmvq_safe = 2 if row > 5120 else 6
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], row, dtype=x.dtype, device=x.device)
    if x.shape[0] <= mmvq_safe:
        return ops.ggml_mul_mat_vec_a8_soa(quants, scales, x, row)
    return ops.ggml_mul_mat_a8(qweight, x, qweight_type, row)


def _fused_mul_mat_gguf_soa_fake(
    x: torch.Tensor,
    quants: torch.Tensor,
    scales: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    row: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], row, dtype=x.dtype, device=x.device)'''
new_soa='''def _fused_mul_mat_gguf_soa(
    x: torch.Tensor,
    quants: torch.Tensor,
    scales: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    row: int,
    force_soa: bool,
) -> torch.Tensor:
    # Maxwell SoA Q4_0 path: 128-bit coalesced quant loads for the decode
    # (vec) kernel. With force_soa (decode-only mode, AoS copy freed) ALL batch
    # sizes use the SoA kernel; otherwise prefill/large batch uses AoS MMQ.
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], row, dtype=x.dtype, device=x.device)
    mmvq_safe = 2 if row > 5120 else 6
    if force_soa or x.shape[0] <= mmvq_safe:
        return ops.ggml_mul_mat_vec_a8_soa(quants, scales, x, row)
    return ops.ggml_mul_mat_a8(qweight, x, qweight_type, row)


def _fused_mul_mat_gguf_soa_fake(
    x: torch.Tensor,
    quants: torch.Tensor,
    scales: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    row: int,
    force_soa: bool,
) -> torch.Tensor:
    return torch.empty(x.shape[0], row, dtype=x.dtype, device=x.device)'''
assert s.count(old_soa)==1, f"old_soa={s.count(old_soa)}"
s=s.replace(old_soa, new_soa, 1)

# 3. _maybe_build_q4_0_soa: free AoS copy in decode-only mode
old_build='''    quants = wb[:, :, 2:18].contiguous().view(nrows, bpr * 16)
    scales = wb[:, :, 0:2].contiguous().view(nrows, bpr * 2).view(torch.float16)
    layer.qweight_soa_quants = quants
    layer.qweight_soa_scales = scales.contiguous()'''
new_build='''    quants = wb[:, :, 2:18].contiguous().view(nrows, bpr * 16)
    scales = wb[:, :, 0:2].contiguous().view(nrows, bpr * 2).view(torch.float16)
    layer.qweight_soa_quants = quants
    layer.qweight_soa_scales = scales.contiguous()
    if _gguf_decode_only():
        # AoS copy no longer needed (SoA handles every batch size).
        _free_param_data(layer.qweight)
        layer.qweight_soa_row = nrows
        layer.qweight_soa_qtype = int(layer.qweight_type.weight_type)'''
assert s.count(old_build)==1, f"old_build={s.count(old_build)}"
s=s.replace(old_build, new_build, 1)

open(F,"w").write(s)
print("patched part1 (helpers, op force_soa, non-sharded free)")

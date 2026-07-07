#!/usr/bin/env python3
"""Idempotent: wire SoA Q4_0 MMVQ into vllm/.../quantization/gguf.py.

- Add _fused_mul_mat_gguf_soa custom op: small-batch (decode) -> SoA mmvq;
  large-batch (prefill) -> falls back to MMQ on the original AoS qweight.
- In process_weights_after_loading, for non-sharded Q4_0 layers, build the
  SoA quants/scales tensors (same bytes, reorganized) and stash on the layer.
- In apply(), non-sharded branch routes to the SoA op when available.
"""
F="/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/layers/quantization/gguf.py"
s=open(F).read()
if "_fused_mul_mat_gguf_soa" in s:
    print("already"); raise SystemExit

# --- 1. Add the SoA op right after _fused_mul_mat_gguf's custom-op registration block ---
reg_anchor='''try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf

except AttributeError as error:
    raise error'''

soa_block='''try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf

except AttributeError as error:
    raise error


def _fused_mul_mat_gguf_soa(
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
    return torch.empty(x.shape[0], row, dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf_soa",
        op_func=_fused_mul_mat_gguf_soa,
        fake_impl=_fused_mul_mat_gguf_soa_fake,
    )
    fused_mul_mat_gguf_soa = torch.ops.vllm._fused_mul_mat_gguf_soa

except AttributeError as error:
    raise error


def _maybe_build_q4_0_soa(layer: torch.nn.Module) -> None:
    """Build SoA quants/scales for a non-sharded Q4_0 qweight, stash on layer."""
    qweight = layer.qweight
    if getattr(qweight, "shard_id", None):
        return  # sharded (qkv/merged) layers stay on the AoS path
    if int(layer.qweight_type.weight_type) != int(WeightType.Q4_0):
        return
    w = qweight.data
    nrows = w.shape[0]
    # each Q4_0 block = 18 bytes (2-byte half scale + 16 quant bytes)
    bpr = w.shape[1] // 18
    wb = w.view(nrows, bpr, 18)
    quants = wb[:, :, 2:18].contiguous().view(nrows, bpr * 16)
    scales = wb[:, :, 0:2].contiguous().view(nrows, bpr * 2).view(torch.float16)
    layer.qweight_soa_quants = quants
    layer.qweight_soa_scales = scales.contiguous()


def _has_q4_0_soa(layer: torch.nn.Module) -> bool:
    return getattr(layer, "qweight_soa_quants", None) is not None'''

assert s.count(reg_anchor)==1, f"reg anchor={s.count(reg_anchor)}"
s=s.replace(reg_anchor, soa_block, 1)

# --- 2. Build SoA in process_weights_after_loading ---
pw_anchor='''        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)'''
pw_new='''        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)
        # Maxwell: build SoA Q4_0 buffers for the decode mmvq fast path.
        _maybe_build_q4_0_soa(layer)'''
assert s.count(pw_anchor)==1, f"pw anchor={s.count(pw_anchor)}"
s=s.replace(pw_anchor, pw_new, 1)

# --- 3. Route in apply() non-sharded branch ---
apply_anchor='''        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf(x, qweight, qweight_type)'''
apply_new='''        else:
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
assert s.count(apply_anchor)==1, f"apply anchor={s.count(apply_anchor)}"
s=s.replace(apply_anchor, apply_new, 1)

open(F,"w").write(s)
print("patched gguf.py")

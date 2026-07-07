#!/usr/bin/env python3
"""Idempotent: add ggml_mul_mat_vec_a8_soa fake + wrapper to _custom_ops.py."""
F="/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/_custom_ops.py"
s=open(F).read()
if "ggml_mul_mat_vec_a8_soa" in s:
    print("already"); raise SystemExit

# 1. fake registration after a8 fake
fake_anchor='''    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)'''
fake_new=fake_anchor+'''

    @register_fake("_C::ggml_mul_mat_vec_a8_soa")
    def _ggml_mul_mat_vec_a8_soa_fake(
        quants: torch.Tensor,
        scales: torch.Tensor,
        X: torch.Tensor,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=quants.device)'''
assert s.count(fake_anchor)==1, f"fake anchor={s.count(fake_anchor)}"
s=s.replace(fake_anchor, fake_new, 1)

# 2. wrapper after a8 wrapper
wrap_anchor='''def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)'''
wrap_new=wrap_anchor+'''


def ggml_mul_mat_vec_a8_soa(
    quants: torch.Tensor,
    scales: torch.Tensor,
    X: torch.Tensor,
    row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8_soa(quants, scales, X, row)'''
assert s.count(wrap_anchor)==1, f"wrap anchor={s.count(wrap_anchor)}"
s=s.replace(wrap_anchor, wrap_new, 1)

open(F,"w").write(s)
print("patched _custom_ops.py")

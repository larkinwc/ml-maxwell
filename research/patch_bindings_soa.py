#!/usr/bin/env python3
"""Idempotent: register ggml_mul_mat_vec_a8_soa in ops.h + torch_bindings.cpp."""
ROOT="/home/larkinwc/maxwell-stack/vllm-maxwell-core/csrc/libtorch_stable"

# 1. ops.h decl
H=f"{ROOT}/ops.h"
s=open(H).read()
if "ggml_mul_mat_vec_a8_soa" not in s:
    anchor="""torch::stable::Tensor ggml_mul_mat_vec_a8(torch::stable::Tensor W,
                                          torch::stable::Tensor X, int64_t type,
                                          int64_t row);"""
    decl=anchor+"""

torch::stable::Tensor ggml_mul_mat_vec_a8_soa(torch::stable::Tensor quants,
                                              torch::stable::Tensor scales,
                                              torch::stable::Tensor X,
                                              int64_t row);"""
    assert s.count(anchor)==1, f"ops.h anchor count={s.count(anchor)}"
    s=s.replace(anchor, decl, 1)
    open(H,"w").write(s)
    print("ops.h patched")
else:
    print("ops.h already")

# 2. torch_bindings.cpp def + impl
B=f"{ROOT}/torch_bindings.cpp"
b=open(B).read()
if "ggml_mul_mat_vec_a8_soa" not in b:
    def_anchor='''  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");'''
    def_new=def_anchor+'''

  // SoA Q4_0 mmvq kernel for Maxwell (128-bit coalesced quant loads).
  ops.def(
      "ggml_mul_mat_vec_a8_soa(Tensor quants, Tensor scales, Tensor X, "
      "SymInt row) -> Tensor");'''
    assert b.count(def_anchor)==1, f"def anchor count={b.count(def_anchor)}"
    b=b.replace(def_anchor, def_new, 1)

    impl_anchor='  ops.impl("ggml_mul_mat_vec_a8", TORCH_BOX(&ggml_mul_mat_vec_a8));'
    impl_new=impl_anchor+'\n  ops.impl("ggml_mul_mat_vec_a8_soa", TORCH_BOX(&ggml_mul_mat_vec_a8_soa));'
    assert b.count(impl_anchor)==1, f"impl anchor count={b.count(impl_anchor)}"
    b=b.replace(impl_anchor, impl_new, 1)
    open(B,"w").write(b)
    print("torch_bindings.cpp patched")
else:
    print("torch_bindings.cpp already")

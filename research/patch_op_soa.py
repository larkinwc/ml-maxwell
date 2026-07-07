#!/usr/bin/env python3
"""Idempotent: add ggml_mul_mat_vec_a8_soa op to gguf_kernel.cu.

Signature: (quants, scales, X, row) -> Y[vecs,row].
quants: uint8[row*bpr*16] flat (or [row,bpr*16]); scales: half[row*bpr].
Quantizes X to q8_1 (same as a8) then runs the SoA Q4_0 kernel. Q4_0 only.
"""
F="/home/larkinwc/maxwell-stack/vllm-maxwell-core/csrc/libtorch_stable/quantization/gguf/gguf_kernel.cu"
MARKER="ggml_mul_mat_vec_a8_soa"
src=open(F).read()
if MARKER in src:
    print("already patched"); raise SystemExit

# insert right before ggml_mul_mat_a8 definition
anchor="torch::stable::Tensor ggml_mul_mat_a8(torch::stable::Tensor W,  // quant weight"

newop = '''torch::stable::Tensor ggml_mul_mat_vec_a8_soa(
    torch::stable::Tensor quants,  // uint8 SoA quant bytes, row*bpr*16
    torch::stable::Tensor scales,  // half SoA scales, row*bpr
    torch::stable::Tensor X,       // input activations [vecs, col]
    int64_t row) {
  int col = X.sizes()[1];
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto Y = torch::stable::empty({vecs, row}, X.scalar_type(), std::nullopt,
                                quants.device());
  cudaStream_t stream = get_current_cuda_stream();
  auto quant_X = torch::stable::empty({vecs, padded / 32 * 9},
                                      torch::headeronly::ScalarType::Int,
                                      std::nullopt, quants.device());
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_mul_mat_vec_a8_soa", [&] {
        quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                         (void*)quant_X.data_ptr(), col, vecs,
                                         stream);
        mul_mat_vec_q4_0_soa_cuda<scalar_t>(
            (void*)quants.data_ptr(), (void*)scales.data_ptr(),
            (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs,
            stream);
      });
  return Y;
}

'''

assert src.count(anchor)==1, f"anchor count={src.count(anchor)}"
src=src.replace(anchor, newop+anchor, 1)
open(F,"w").write(src)
print("patched gguf_kernel.cu")

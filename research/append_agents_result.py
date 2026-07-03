#!/usr/bin/env python3
F="/home/larkinwc/maxwell-stack/AGENTS.md"
MARKER="## SoA Q4_0 MMVQ kernel — SHIPPED"
src=open(F).read()
if MARKER in src:
    print("already present"); raise SystemExit
block='''
## SoA Q4_0 MMVQ kernel — SHIPPED (+5.5% end-to-end)

Implemented the handwritten SoA (Structure-of-Arrays) Q4_0 decode kernel from the
microbench above. **Committed on `maxwell/v0.23`.**

**What it does:** at load time, repack each 18-byte Q4_0 block into two contiguous
arrays — `quants` uint8[nrows*bpr*16] (block-contiguous, 16B aligned, loaded as `uint4`
= 128-bit coalesced) + `scales` half[nrows*bpr]. New op `ggml_mul_mat_vec_a8_soa`
(quants, scales, X, row) runs the SoA kernel `mul_mat_vec_q4_0_soa` (MMV_Y=2, one thread
per block, warp strides 32 blocks). Math is bit-identical to `vec_dot_q4_0_q8_1_impl`
(validated: SoA vs AoS op max_abs ~1e-5).

**Files touched (all on `maxwell/v0.23`):**
- `csrc/.../gguf/mmvq.cuh` — kernel `mul_mat_vec_q4_0_soa` + `mul_mat_vec_q4_0_soa_cuda` launcher.
- `csrc/.../gguf/gguf_kernel.cu` — op `ggml_mul_mat_vec_a8_soa` (quantizes X to q8_1, runs SoA kernel).
- `csrc/.../ops.h`, `torch_bindings.cpp` — decl + def/impl registration.
- `vllm/_custom_ops.py` — python wrapper + fake.
- `vllm/.../quantization/gguf.py` — `_fused_mul_mat_gguf_soa` op, `_maybe_build_q4_0_soa`
  (non-sharded) + per-shard SoA build in `_create_padded_weight_param` (qkv/gate_up),
  routing in `apply()`. Env escape hatch `VLLM_DISABLE_Q4_0_SOA=1` falls back to AoS.

**Measured (Qwen3.5-9B Q4_0, TP=4, graphs on, 128 tok, deterministic 2 runs):**

| config | tok/s |
|--------|-------|
| baseline (AoS) | 23.10 |
| **full SoA** | **24.38 (×2 identical)** |

**= +5.5% end-to-end.** Output coherent (Paris). Why not the full +19%: MMVQ is ~62% of
decode; 0.62 × 0.19 ≈ +12% on the matmul, diluted by the GDN-SSM/attention/sampling ops
(~38%) that don't benefit → ~+5.5% net. Coverage matters: non-sharded layers alone (40% of
Q4_0 bytes) gave only +1.9% at TP=2; extending SoA to the sharded qkv/gate_up layers (the
other 60%) is what lifted it to +5.5%.

**Memory cost:** SoA buffers are kept ALONGSIDE the AoS qweight (prefill/MMQ still needs AoS),
so Q4_0 weight memory ~doubles. 9B does NOT fit TP=1 with SoA on; validate at TP≥2 (TP=4 used
above). Future: free the AoS copy for decode-only deployments, or repack in place.

- **Q4_0 byte layout (verified):** block = [half scale (2B)][16 quant bytes]; lo nibble→elem
  0..15, hi nibble→elem 16..31; dequant `(nibble-8)*scale`. Matches gguf.dequantize exactly.
'''
open(F,"a").write(block)
print("appended result")

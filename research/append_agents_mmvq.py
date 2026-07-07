#!/usr/bin/env python3
F="/home/larkinwc/maxwell-stack/AGENTS.md"
MARKER="## MMVQ decode kernel: where the bandwidth actually goes"
src=open(F).read()
if MARKER in src:
    print("already present"); raise SystemExit

block = '''
## MMVQ decode kernel: where the bandwidth actually goes

The GGUF decode hot path is `mul_mat_vec_q` (`csrc/.../gguf/mmvq.cuh`), 62% of decode
self-CUDA time. It is **HBM-bandwidth-bound**, NOT ALU-bound and NOT latency-bound.
Three experiments nailed this down (Qwen3.5-9B Q4_0, TP=1, graphs on):

1. **Register-only software `__dp4a`** (replaced the `(const int8_t*)&a` byte-load emulation
   in `vecdotq.cuh` with address-free arithmetic-shift sign-extension): **0% change**
   (MMVQ self-time 6.420s vs 6.414s; 7.60 vs 7.56 tok/s). The dp4a math fully overlaps with
   weight loads. **REVERTED** — llama.cpp's own sm_50 fallback is byte-identical anyway, so
   there is no better reference to port. Optimizing the quant ALU is a dead end on GM10x.

2. **Occupancy sweep `GGML_CUDA_MMV_Y` ∈ {1,2,4,8}** (rows per block / warps per block):
   1→7.56, 2→7.71, 4→7.69, 8→7.67 tok/s. **+2% then immediate plateau.** Enough warps are
   already resident to hide HBM latency. Set MMV_Y=2 if you want the free 2%, but it is noise.

3. **Achievable HBM bandwidth (microbench, one die):** copy 72.0 GB/s, read-only 73.1 GB/s.
   The M10 spec ~88 GB/s, so ~82% is the practical ceiling. Use **73 GB/s as the ceiling.**

**The real gap = MMVQ load efficiency.** Isolated `ggml_mul_mat_vec_a8` over all real 9B
Q4_0 weights achieves only **43 GB/s = 59% of 73**. A faithful standalone CUDA harness
(`q40_bw2.cu`, mirrors `vec_dot_q4_0_q8_1` incl. q8_1 y-reads) reproduces it: **AoS 48.7 GB/s
(67%)**. The cause is the **18-byte `block_q4_0` AoS layout** (`half d` + 16 quant bytes):
strided non-power-of-2 blocks straddle 128-byte lines → partial/uncoalesced sectors, and the
2-byte scale prefix prevents 16-byte-aligned 128-bit loads.

**SoA repack win (measured, `q40_bw2.cu`):** split each block into a contiguous quant array
(16B-aligned, loaded as `uint4`/128-bit) + a separate contiguous scale array:

| layout | MMV_Y=1 | MMV_Y=2 | MMV_Y=4 |
|--------|---------|---------|---------|
| AoS (current) | 48.7 GB/s (67%) | 50.7 (69%) | 50.4 (69%) |
| **SoA int4 128-bit** | **58.3 GB/s (80%)** | 57.8 (79%) | 57.8 (79%) |

**+19% kernel bandwidth (48.7→58.3), 67%→80% of peak.** Projected end-to-end: 62% of decode
× 1.19 ≈ **~11% total tok/s** (TP=1 ~7.6→~8.4). Requires a load-time Q4_0 repack (scales/quants
into two arrays) + a new SoA MMVQ kernel + dispatch. Microbench sources live in
`~/maxwell-stack/q40_bw.cu`, `q40_bw2.cu`, `mmvq_bw.py`, `mmvq_single.py`, `bw_test.py`.

- **ncu / Nsight Compute does NOT support Maxwell sm_50** ("Profiling is not supported on
  device 0"); nvprof is absent. No HW-counter profiling on this box — use microbenchmarks +
  the torch profiler (op self-CUDA time) instead. `RmProfilingAdminOnly:1` but we have
  passwordless sudo; irrelevant since ncu rejects the arch regardless.
'''

open(F,"a").write(block)
print("appended")

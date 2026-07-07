#!/usr/bin/env python3
F="/home/larkinwc/maxwell-stack/AGENTS.md"
MARKER="## SoA Q4_0 — TP scaling + decode-only mode"
src=open(F).read()
if MARKER in src:
    print("already"); raise SystemExit
block='''
## SoA Q4_0 — TP scaling + decode-only mode

**TP scaling of the SoA win (9B Q4_0, graphs on, 128 tok):**

| TP | baseline (AoS) | SoA | gain |
|----|----------------|-----|------|
| 1  | 7.58 | **8.30** (decode-only) | **+9.5%** |
| 4  | 23.10 | 24.38 | +5.5% |
| 8  | 29.48 | 30.27 | +2.7% |

The win shrinks as TP grows: more dies => less MMVQ work per die + the host-staged
NCCL allreduce is a bigger share of decode time, diluting the kernel gain. **TP=1 is the
biggest win** (MMVQ dominates decode, no allreduce).

**Decode-only mode `VLLM_GGUF_DECODE_ONLY=1` (committed):**
- The default SoA path keeps BOTH the AoS qweight (for prefill/MMQ) and the SoA buffers,
  ~doubling Q4_0 weight memory (+3.5 GB model-wide for 9B). 9B therefore OOMs at TP=1 with
  SoA on.
- The SoA op `ggml_mul_mat_vec_a8_soa` is correct for ALL batch sizes (validated vecs
  1..32 vs AoS, max_abs ~1e-3), just slower than MMQ for big prefill batches.
- So with `VLLM_GGUF_DECODE_ONLY=1` we FREE the AoS copy after building SoA (set
  `param.data = empty(0)`) and route every batch through SoA. This:
  - Makes 9B SoA FIT at TP=1 (8.30 tok/s, +9.5% vs 7.58 baseline) — previously OOM.
  - Recovers ~0.9 GB/die at TP=4 -> KV cache 131K -> 154,688 tokens (+18% concurrency),
    same speed (24.40 vs 24.38).
  - Intended for the M10 decode tier (prefill runs on the MI100s). NOT for a node that
    must also prefill large batches fast (SoA-as-prefill is slower than MMQ).
- **Gotcha fixed:** `token_embd` is a non-sharded Q4_0 too, but the embedding path uses
  `ggml_dequantize` and reads `qweight.shape[1]` directly. `GGUFEmbeddingMethod` now
  overrides `process_weights_after_loading` to SKIP SoA build/free (else IndexError on the
  freed empty qweight during profile_run).
- **Flags:** `VLLM_DISABLE_Q4_0_SOA=1` now also skips BUILDING the SoA buffers (true AoS
  baseline, no extra memory); it is ignored when `VLLM_GGUF_DECODE_ONLY=1` (SoA required).
'''
open(F,"a").write(block)
print("appended")

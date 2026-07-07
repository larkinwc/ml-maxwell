# Status / build & validation checklist

Live tracker for the Maxwell decode tier. Updated as artifacts land.

## Decode performance (2026-07-07, evo hillclimb — see ROADMAP.md update)

| Piece | State |
|---|---|
| **Batch-8 decode TP=4+graphs** | ✅ **49.0 tok/s** (was 18.5 — 2.65×); b16 50.3, b64 105.2, b128 122.2; single-stream 17.3 (was 4.4) |
| Fused GGUF kernels on sm_50 | ✅ dp4a fallback + guard fix (vecdotq bodies compiled EMPTY < cc 6.1); v2/v3 sidecar kernels numerics-gated (max_rel ≤ 1e-3) |
| Plain-quant in_proj (no --f16-inproj) | ✅ **FIXED** — vLLM merged-shard concat used GGUF file order; `sorted(shard_id)` + same-type grouping. All-quant FIXED.gguf is the batch champion |
| Kernel dispatch | v1 MMVQ (b1) / v3 FFMA (b2–16) / dequant+cuBLAS (b>16 + prefill — **MMQ dropped**: ≡ dequant at decode, 2× slower at prefill); env-gated via MAXWELL_EVO_* (sidecar: `tools/maxwell/evo/`) |
| Prefill / TTFT | ✅ 86 tok/s, 5.9 s @512 (was 44 / 11.7 s). Next wall: torch-native GDN chunk scan → llama.cpp `gated_delta_net.cu` port (planned, journal has the spec) |
| `_C` rebuild with kernel fixes | 🔄 in progress on the box (2026-07-07) — bakes the dp4a/vecdotq fixes into `_C_stable_libtorch`; smoke test after: `MAXWELL_GGUF_DEQUANT=0` without sidecar must be coherent |
| Fork PRs | ✅ opened: [vllm-maxwell-core#11](https://github.com/larkinwc/vllm-maxwell-core/pull/11) (kernels+fixes), [ml-maxwell#1](https://github.com/larkinwc/ml-maxwell/pull/1) (docs) |
| Upstream vLLM PRs (2 gguf.py fixes) | ⏳ candidates flagged — needs human owner per vLLM AGENTS.md |
| DP replicas / TP=16 | ⛔ measured dead ends (host-bus saturation / fixed-floor dominance) |

## Native floor (the slow, stable layer)

| Artifact | State | Notes |
|---|---|---|
| NCCL 2.20.5 sm_50 | ✅ **built & proven** | all-reduce passes 2/4/8 GPU on tyangpu1; `.so` rescued to `~/maxwell-stack/artifacts` |
| llama.cpp sm_50 | ✅ built | reference/fallback decode |
| **torch 2.11 sm_50** | ✅ **BUILT & ON-DEVICE VALIDATED** | `torch-2.11.0a0+git70d99e9-cp312` (293MB, sha256 aefee046…). Compiled clean for sm_50 (zero CUDA-arch errors — no sm_70 guard broke). fp16 matmul + reduction verified correct on a real Tesla M10 (cc 5.0). The session gate is CLEARED. |

## vLLM v0.23 fork (`vllm-maxwell-core`, branch `maxwell/v0.23`)

| Piece | State |
|---|---|
| **Full CUDA build for sm_50;sm_52** | ✅ **ALL 301 TARGETS COMPILE + LINK** — vllm installed (`0.1.dev17369+g0f315eb82.cu126`); `_C.abi3.so` carries sm_50+sm_52 SASS; `vllm._C` imports |
| **paged_attention_v1 on real M10** | ✅ **ON-DEVICE NUMERICS PASS** — max abs err 3e-4 vs fp32 ref (pure fp16 rounding); runs correctly on hardware with no fp16 ALU |
| CMake arch gate (5.0;5.2) | ✅ commit `4cb64bb` |
| fp16 emulation — attention `dtype_float16.cuh` | ✅ commit `0a58233` (5 leaf add/mul/fma primitives, convert→fp32→convert) |
| fp16 emulation — moe `moe_wna16_utils.h` | ✅ commit `0a58233` (maxwell_safe_hsub2/hfma2) |
| fp16 emulation — gptq `compat.cuh` | ✅ commit `0a58233` (macro-routed __hadd/__hmul/__hsub/__hfma + CUDA 12.6 atomicAdd compat fix) |
| fp16 emulation — awq `gemm_kernels.cu` | ✅ commit `0f315eb` (awq_sub_h2/awq_fma_h2 in dequant_weights) |
| `MaxwellCommunicator` (NCCL 2.20.5) | ✅ scaffold (thin subclass) |
| Quant (GPTQ/AWQ/WNA16) | ✅ compiles for sm_50 — on-device numerics test pending |

> **The fp16-arithmetic wall is CLEARED.** Maxwell GM10x has fp16 storage+convert
> but no fp16 ALU (add.f16/fma.f16x2 need sm_53+). Mirroring ggml/llama.cpp, all
> fp16 math is emulated convert→fp32→compute→convert. Only 4 source families
> needed patching (attention, gptq, awq, moe-wna16); the other 297 targets were
> already sm_50-clean. See `FP16_ON_MAXWELL.md`.

## Critical path

1. ✅ **Build torch-sm50** (torch 2.11 on CUDA 12.6) → wheel built + validated on M10. *gate cleared.*
2. ✅ `vllm-maxwell` conda env (Py3.12) on tyangpu1; torch-sm50 installed & smoke-tested.
3. ✅ Publish torch-sm50 wheel to GH Releases.
4. ✅ **Build/install vLLM v0.23** (from `vllm-maxwell-core` fork) — all 301 CUDA targets
   compile + link for sm_50;sm_52; the fp16-arithmetic wall cleared via fp32 emulation.
5. ✅ **On-device numerics test**: `paged_attention_v1` on a real M10 → max abs err 3e-4. PASS.
6. Numerics-test the rest: `paged_attention_v2`, gptq/awq dequant, moe_wna16.
7. Runtime deps for serving: install `zmq` etc. (built `--no-deps`); get `vllm` python
   import clean (currently blocked only by missing runtime pkgs, not the C ext).
8. ✅ **End-to-end generation** — exceeded: Qwen3.5-9B GGUF (hybrid GDN), coherent
   output at TP=2/4/8/16 on the C4130 M10s (branch `maxwell/qwen35-gguf-sm50`,
   see `vllm-maxwell-core/tools/maxwell/README.md`).
9. ✅ **CUDA graphs** — FULL capture auto-downgrades to FULL_DECODE_ONLY for the GDN
   backend; ~1.5× decode vs eager (18.5 tok/s TP=4 on Qwen3.5-9B).
10. (optional) Rust `vllm-server` gRPC binary needs `protoc` — non-fatal, skip unless needed.
11. Next: performance tier — see `docs/ROADMAP.md`.

> Historical (tyangpu1 bring-up box): **device_count = 12** (8 M10 dies + 4 MI100);
> scope decode via `CUDA_VISIBLE_DEVICES`. Current perf box (2026-07):
> **uno-PowerEdge-C4130** — 4× Tesla M10 = 16 dies, 1× E5-2695 v4 (single NUMA),
> 4× DDR4-2133 @ 1DPC (all 4 channels, optimal), Intel DC SSD root, no swap.

## Hardware levers (optional, cheap)

- ~~Populate empty DIMM channels~~ — done differently per box: the C4130 already
  runs 1DPC on all 4 channels of its single socket (A1–A4); extra DIMMs there add
  capacity only. On tyangpu1 (2→4 channels) it remains a ~2× all-reduce-ceiling lever.

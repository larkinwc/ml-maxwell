# Status / build & validation checklist

Live tracker for the Maxwell decode tier. Updated as artifacts land.

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
8. End-to-end: load a small fp16 model, generate tokens, TP=2 across two dies.
9. Enable piecewise cudagraphs (CUDA 12.6 supports sm_50 graph capture).
10. (optional) Rust `vllm-server` gRPC binary needs `protoc` — non-fatal, skip unless needed.

> **device_count = 12** on this box (8 M10 dies + 4 MI100). Scope decode to the
> M10s via `CUDA_VISIBLE_DEVICES` to keep the tiers separate.

## Hardware levers (optional, cheap)

- Populate empty DIMM channels (2→4 per socket) → ~2× host RAM BW (~37 GB/s) →
  ~2× host-staged all-reduce ceiling. ~$10–15/DIMM used.

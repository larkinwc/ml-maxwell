# Status / build & validation checklist

Live tracker for the Maxwell decode tier. Updated as artifacts land.

## Native floor (the slow, stable layer)

| Artifact | State | Notes |
|---|---|---|
| NCCL 2.20.5 sm_50 | ✅ **built & proven** | all-reduce passes 2/4/8 GPU on tyangpu1; `.so` rescued to `~/maxwell-stack/artifacts` |
| llama.cpp sm_50 | ✅ built | reference/fallback decode |
| **torch 2.11 sm_50** | ✅ **BUILT & ON-DEVICE VALIDATED** | `torch-2.11.0a0+git70d99e9-cp312` (293MB, sha256 aefee046…). Compiled clean for sm_50 (zero CUDA-arch errors — no sm_70 guard broke). fp16 matmul + reduction verified correct on a real Tesla M10 (cc 5.0). The session gate is CLEARED. |

## vllm-maxwell plugin (plugin-first vs unmodified upstream v0.23)

| Piece | State |
|---|---|
| Package + entry point (`vllm.platform_plugins`) | ✅ scaffold |
| `MaxwellPlatform` (v0.23 interface, `CudaPlatformBase`) | ✅ scaffold, real hooks stubbed |
| `MaxwellAttentionBackend` (csrc FMA paged-attn) | ⏳ skeleton — needs torch-sm50 to validate |
| `MaxwellCommunicator` (NCCL 2.20.5) | ✅ scaffold (thin subclass) |
| Quant (GPTQ/AWQ legacy dequant, 4-bit) | ⏳ deferred (fp16 first) |
| vLLM fork (inside ml-maxwell) | ⛔ not created — only if plugin hits a hard wall |

## Critical path

1. ✅ **Build torch-sm50** (torch 2.11 on CUDA 12.6) → wheel built + validated on M10. *gate cleared.*
2. ✅ `vllm-maxwell` conda env (Py3.12) on tyangpu1; torch-sm50 installed & smoke-tested.
3. Publish torch-sm50 wheel to GH Releases + Pages index.
4. Build/install **vLLM v0.23** (from the `vllm-maxwell-core` fork) against this torch.
5. **On-device numerics test**: run csrc `paged_attention_v1` on a real M10 tensor.
   *Make-or-break for attention* — proves the FMA paged-attn path computes correctly.
6. Identify the **minimal capability-gated fork diff** (sm_50 build gates, dtype floors,
   attn-backend selector) needed to get v0.23 importing/running on Maxwell.
7. Wire `MaxwellAttentionBackend` fully (Impl + MetadataBuilder, v0.23 interface).
8. End-to-end: load a small fp16 model, generate tokens, TP=2 across two dies.
9. Enable piecewise cudagraphs (CUDA 12.6 supports sm_50 graph capture).
10. Quant kernels for 4-bit (fit bigger models in 8 GB).

> **device_count = 12** on this box (8 M10 dies + 4 MI100). Scope decode to the
> M10s via `CUDA_VISIBLE_DEVICES` to keep the tiers separate.

## Hardware levers (optional, cheap)

- Populate empty DIMM channels (2→4 per socket) → ~2× host RAM BW (~37 GB/s) →
  ~2× host-staged all-reduce ceiling. ~$10–15/DIMM used.

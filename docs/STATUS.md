# Status / build & validation checklist

Live tracker for the Maxwell decode tier. Updated as artifacts land.

## Native floor (the slow, stable layer)

| Artifact | State | Notes |
|---|---|---|
| NCCL 2.20.5 sm_50 | ✅ **built & proven** | all-reduce passes 2/4/8 GPU on tyangpu1; `.so` rescued to `~/maxwell-stack/artifacts` |
| llama.cpp sm_50 | ✅ built | reference/fallback decode |
| **torch 2.9.1 sm_50** | ⏳ **GATE — not built** | blocks every on-device runtime test; CI workflow ready (`build-torch-sm50.yml`) |
| torch 2.10 sm_50 | 🔭 stretch | would unlock vLLM v0.19; verify it source-builds sm_50 |

## vllm-maxwell plugin

| Piece | State |
|---|---|
| Package + entry point (`vllm.platform_plugins`) | ✅ scaffold |
| `MaxwellPlatform` (v0.16 interface) | ✅ scaffold, real hooks stubbed |
| `MaxwellAttentionBackend` (csrc FMA paged-attn) | ⏳ skeleton — needs torch-sm50 to validate |
| `MaxwellCommunicator` (NCCL 2.20.5) | ✅ scaffold (thin subclass) |
| Quant (GPTQ/AWQ legacy dequant, 4-bit) | ⏳ deferred (fp16 first) |

## Critical path

1. **Build torch-sm50** (CI, GPU-free) → publish wheel. ← *the gate*
2. Stand up `vllm-maxwell` conda env (Py3.12) on tyangpu1; install torch-sm50.
3. **On-device numerics test**: run csrc `paged_attention_v1` on a real M10 tensor.
   *Make-or-break* — proves the FMA attention path computes correctly.
4. Wire `MaxwellAttentionBackend` fully (Impl + MetadataBuilder, v0.16 interface).
5. End-to-end: load a small fp16 model, generate tokens, TP=2 across two dies.
6. Enable piecewise cudagraphs (CUDA 12.6 supports sm_50 graph capture).
7. Quant kernels for 4-bit (fit bigger models in 8 GB).

## Hardware levers (optional, cheap)

- Populate empty DIMM channels (2→4 per socket) → ~2× host RAM BW (~37 GB/s) →
  ~2× host-staged all-reduce ceiling. ~$10–15/DIMM used.

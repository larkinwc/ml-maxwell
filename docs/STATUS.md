# Status / build & validation checklist

Live tracker for the Maxwell decode tier. Updated as artifacts land.

## Native floor (the slow, stable layer)

| Artifact | State | Notes |
|---|---|---|
| NCCL 2.20.5 sm_50 | ✅ **built & proven** | all-reduce passes 2/4/8 GPU on tyangpu1; `.so` rescued to `~/maxwell-stack/artifacts` |
| llama.cpp sm_50 | ✅ built | reference/fallback decode |
| **torch 2.11 sm_50** | ⏳ **GATE — not built** | built on CUDA 12.6; blocks every on-device runtime test; CI ready (`build-torch-sm50.yml`). Risk: an sm_70+ kernel guard could break the build → fall back to 2.10 |

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

1. **Build torch-sm50** (CI, GPU-free, torch 2.11 on CUDA 12.6) → publish wheel. ← *the gate*
2. Stand up `vllm-maxwell` conda env (Py3.12) on tyangpu1; install torch-sm50 + vLLM v0.23.
3. **On-device numerics test**: run csrc `paged_attention_v1` on a real M10 tensor.
   *Make-or-break* — proves the FMA attention path computes correctly.
4. **Plugin smoke test against unmodified v0.23** — if it hits a hard cu13/sm_70
   wall, fork vLLM inside ml-maxwell with a minimal capability-gated diff.
5. Wire `MaxwellAttentionBackend` fully (Impl + MetadataBuilder, v0.23 interface).
6. End-to-end: load a small fp16 model, generate tokens, TP=2 across two dies.
7. Enable piecewise cudagraphs (CUDA 12.6 supports sm_50 graph capture).
8. Quant kernels for 4-bit (fit bigger models in 8 GB).

## Hardware levers (optional, cheap)

- Populate empty DIMM channels (2→4 per socket) → ~2× host RAM BW (~37 GB/s) →
  ~2× host-staged all-reduce ceiling. ~$10–15/DIMM used.

# ml-maxwell

Build + distribution infrastructure for running modern ML inference on **NVIDIA Maxwell
(sm_50) GPUs** — specifically **Tesla M10** boards (`GM107`, 8 GB/die, no tensor cores,
no GPU P2P).

This repo produces prebuilt, sm_50-capable binary artifacts (the slow, stable native
floor) and hosts the **`vllm-maxwell`** out-of-tree platform plugin that lets *upstream*
vLLM run a decode tier on Maxwell hardware.

## Why this exists

Maxwell is past end-of-life in the modern CUDA/PyTorch toolchain:

| Layer | Maxwell status |
|---|---|
| CUDA 12.6 | sm_50 **supported** (deprecated but emits Maxwell) ✅ |
| CUDA 12.8 / 12.9 | toolkit builds sm_50, but PyTorch's 12.8 *binaries* drop it |
| **CUDA 13.0+** | sm_50 **removed** — `nvcc` cannot emit Maxwell code |
| PyTorch 2.8+ prebuilt wheels (cu128/cu13) | Maxwell kernels **deleted** from binaries |
| **PyTorch source build (any version) on CUDA 12.6** | sm_50 **builds** ✅ — incl. v2.11 |

**The only hard floor is the CUDA 12.6 toolkit.** torch and vLLM can be *latest* —
the Maxwell deletion was from *prebuilt binaries built against CUDA 12.8/13*, NOT
from the torch source tree. Verified: torch **v2.11** cmake still lists
`CUDA_COMMON_GPU_ARCHITECTURES = "3.5" "5.0"` and maps Maxwell→`5.0 5.2`, identical
to 2.9. Build latest torch against CUDA 12.6 → sm_50 kernels come back.

## Pinned stack (the "Maxwell floor")

| Component | Pin | Notes |
|---|---|---|
| **CUDA Toolkit** | **12.6** (HARD FLOOR) | only thing actually pinned; 12.8/13 binaries drop Maxwell, CUDA 13 can't emit sm_50 at all; full CUDA-graph support |
| PyTorch | **2.11.0** (source, `TORCH_CUDA_ARCH_LIST=5.0;5.2`, built on CUDA 12.6) | matches vLLM v0.23; source tree still accepts sm_50 |
| vLLM (decode tier) | **v0.23.x** — minimal-diff fork (`larkinwc/vllm` @ `maxwell/v0.23`, submodule) | diff capability-gated for mechanical rebase |
| Python | **3.12** | |
| NCCL | **2.20.5** (source, sm_50) | predates `cudaMemPoolCreate` dep that breaks 2.30 on Maxwell |

> **Risk to validate at build time:** torch 2.11 cmake *accepts* sm_50, but a newer
> kernel could carry an `#if __CUDA_ARCH__ >= 700` guard that fails to compile for
> Maxwell. The build proves it; the arch is NOT categorically blocked. Fallback:
> torch 2.10 (also 12.6-capable).

> **CUDA graphs work on Maxwell with CUDA 12.6.** The graph API (capture/replay) is
> arch-independent since CUDA 10. No CUDA 11.x needed; 11.x would only lose features.

## Hardware facts (target: host `tyangpu1`)

- 2× Tesla M10 = **8× GM107 dies**, sm_50, ~6.9 GB usable VRAM each.
- **No GPU P2P** (GM107 silicon never implemented peer-DMA) — even the on-card pair.
  TP comms go **host-staged** via NCCL 2.20.5 SHM transport.
- All dies behind PLX PEX 8747 switches on a single root port (NUMA node 1).
- Host RAM bandwidth ~18.8 GB/s (only 2 of 4 channels populated) — the real ceiling
  for host-staged all-reduce. (Cheap fix: populate the empty DIMM channels → ~2×.)

## Architecture

Disaggregated, Mooncake-inspired:

- **Prefill: MI100 (CDNA)** — separate optimized vLLM fork (not in this repo).
- **Decode: Tesla M10 (this repo)** — minimal-diff **vLLM fork** (`larkinwc/vllm`,
  branch `maxwell/v0.23`, embedded here as a submodule) + the `vllm-maxwell` plugin
  package. Fork diff is kept small and capability-gated so rebasing onto each new
  upstream release stays mechanical (preserves day-0 model support).
- KV blocks shipped prefill→decode via Mooncake Transfer Engine (bf16→fp16 cast,
  identical vLLM KV layout, no remap).

## Repo layout

```
ml-maxwell/
├── artifacts/                 # build recipes + manifests for the native floor
│   ├── nccl-2.20.5-sm50/      # proven NCCL build (all-reduce passes 2/4/8 GPU)
│   ├── torch-sm50/            # PyTorch 2.9.1 sm_50 source-build recipe
│   └── llama-cpp-sm50/        # reference/fallback decode engine
├── vllm-maxwell/              # out-of-tree vLLM platform plugin (pip package)
├── vllm-maxwell-core/         # submodule: larkinwc/vllm-maxwell-core @ maxwell/v0.23
│                              #   minimal-diff vLLM fork (clean upstream v0.23 base)
├── .github/workflows/         # CI: GPU-free cross-compile + self-hosted numerics
└── docs/                      # design notes, fork-patch / upstream-PR tracker
```

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/larkinwc/ml-maxwell.git
# or, after a plain clone:
git submodule update --init --recursive
```

## Distribution

Heavy wheels (torch-sm50, etc.) published to **GitHub Releases**; a **GitHub Pages
PEP 503 index** makes them pip-installable:

```bash
pip install vllm-maxwell --extra-index-url https://larkinwc.github.io/ml-maxwell/simple/
```

## Status

Early scaffolding. See `docs/STATUS.md` for the live build/validation checklist.

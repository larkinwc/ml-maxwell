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
| CUDA 12.8 / 12.9 | sm_50 **deprecated** (builds, with warnings) |
| **CUDA 13.0+** | sm_50 **removed** — `nvcc` cannot emit Maxwell code |
| PyTorch 2.8+ prebuilt wheels | Maxwell kernels **deleted** from binaries |
| PyTorch source build on CUDA 12.x | sm_50 **still builds** ✅ |

The consequence: **the native floor is permanently pinned to the CUDA 12.6 / torch 2.9
era** — the last combination that can emit sm_50 *and* is new enough to be useful.
Newer is impossible (CUDA 13 has no sm_50); older loses features for no gain.

## Pinned stack (the "Maxwell floor")

| Component | Pin | Notes |
|---|---|---|
| CUDA Toolkit | **12.6** | last Maxwell-friendly toolkit; full CUDA-graph support |
| PyTorch | **2.9.1** (source, `TORCH_CUDA_ARCH_LIST=5.0;5.2`) | newest torch that source-builds sm_50 |
| vLLM (decode tier) | **v0.16.x** | newest vLLM whose torch pin (2.9.1) supports Maxwell |
| Python | **3.12** | |
| NCCL | **2.20.5** (source, sm_50) | predates `cudaMemPoolCreate` dep that breaks 2.30 on Maxwell |
| Stretch goal | torch 2.10 → vLLM v0.19 | validate 2.10 source-builds sm_50 later |

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
- **Decode: Tesla M10 (this repo)** — `vllm-maxwell` plugin.
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
├── .github/workflows/         # CI: GPU-free cross-compile + self-hosted numerics
└── docs/                      # design notes, upstream-PR tracker
```

## Distribution

Heavy wheels (torch-sm50, etc.) published to **GitHub Releases**; a **GitHub Pages
PEP 503 index** makes them pip-installable:

```bash
pip install vllm-maxwell --extra-index-url https://larkinwc.github.io/ml-maxwell/simple/
```

## Status

Early scaffolding. See `docs/STATUS.md` for the live build/validation checklist.

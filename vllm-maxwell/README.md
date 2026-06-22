# vllm-maxwell

Out-of-tree **vLLM platform plugin** that runs a decode tier on NVIDIA Maxwell
(sm_50) GPUs — Tesla M10.

## Why a plugin and not a fork

The goal is **day-0 support for new model architectures**. A fork would force a
perpetual rebase against upstream and constantly drift from day-0. Instead this is
a separate pip package that registers a Maxwell platform via vLLM's
`vllm.platform_plugins` entry point — **vanilla upstream vLLM + `pip install
vllm-maxwell`**. New models land upstream → we get them for free, because we never
touch core.

The handful of unavoidable sm_50 core touchpoints (capability build-gates, dtype
floors) are contributed as **upstream PRs gated behind capability checks** so they
are no-ops on modern hardware. See `../docs/UPSTREAM_PRS.md`.

## What it overrides

`MaxwellPlatform` subclasses the in-tree `CudaPlatform` and changes only:

| Hook | Maxwell behavior |
|---|---|
| `supported_dtypes` | fp16 + fp32 only (no bf16 / fp8 / tensor cores) |
| `check_and_update_config` | cast bf16→fp16; bring-up uses enforce_eager, then cudagraphs |
| `get_attn_backend_cls` | `MaxwellAttentionBackend` — legacy csrc FMA paged-attention |
| `get_device_communicator_cls` | `MaxwellCommunicator` — host-staged NCCL 2.20.5 (no P2P) |

## Pinned floor (install from the ml-maxwell index, not PyPI)

- torch **2.9.1** — *source-built for sm_50* (PyPI wheels have no Maxwell kernels)
- vLLM **v0.16.x** — newest vLLM whose torch pin still supports Maxwell
- CUDA **12.6**, Python **3.12**
- NCCL **2.20.5** (sm_50, host-staged)

```bash
pip install vllm-maxwell --extra-index-url https://larkinwc.github.io/ml-maxwell/simple/
```

## Status

Scaffold. `platform.py` targets the real v0.16 interface; `attention.py` and
`communicator.py` are skeletons pending the torch-sm50 wheel (the gate that
unblocks the make-or-break on-device numerics test).

# torch-sm50

PyTorch **2.9.1** built from source with Maxwell (**sm_50**) kernels.

## Why

Prebuilt PyTorch wheels (2.8+) **deleted Maxwell kernels** to keep binary size down
([pytorch#157517](https://github.com/pytorch/pytorch/issues/157517)). Running anything
on a Tesla M10 therefore requires a source build. This wheel **is the gate** for the
entire `vllm-maxwell` stack — NCCL bindings, csrc attention, and the plugin all link it.

## Key facts

- **Cross-compiles without a Maxwell GPU.** `nvcc` emits sm_50 cubins on any CUDA 12.6
  host; no device is touched at build time. → CI builds on GitHub-hosted runners.
- **CUDA must be 12.x.** CUDA 13 removed sm_50; `build.sh` hard-fails on CUDA ≥ 13.
- **torch 2.9.1 is the ceiling.** It's the newest torch that still source-builds sm_50
  *and* is pinned by a usable vLLM (v0.16.x). torch 2.10 (vLLM v0.19) is a stretch goal.
- Tensor-core paths (flash-attn, mem-eff-attn, cusparseLt) are **disabled** — Maxwell
  has no tensor cores; decode attention comes from legacy csrc in `vllm-maxwell`.
- Links our proven **NCCL 2.20.5 sm_50** via `USE_SYSTEM_NCCL`.

## Build

```bash
CUDA_HOME=/usr/local/cuda-12.6 \
NCCL_ROOT=/path/to/nccl-2.20.5-sm50 \
PYTHON_BIN=python3.12 \
OUTPUT_DIR=./dist \
./build.sh
```

Output: `torch-2.9.1-cp312-cp312-linux_x86_64.whl` → published to GH Releases + Pages index.

## Validation (requires real M10 — self-hosted runner)

```python
import torch
assert torch.cuda.is_available()
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = x @ x            # exercises sm_50 FMA matmul path
torch.cuda.synchronize()
print("sm_50 matmul OK", y.sum().item())
```

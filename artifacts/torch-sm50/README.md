# torch-sm50

PyTorch **2.11.0** (latest, matches vLLM v0.23) built from source with Maxwell
(**sm_50**) kernels, **against CUDA 12.6**.

## Why

Prebuilt PyTorch wheels (2.8+) **deleted Maxwell kernels** — but only from binaries
built against CUDA 12.8/13 ([pytorch#157517](https://github.com/pytorch/pytorch/issues/157517)).
The **torch source tree still accepts sm_50** at latest: v2.11 cmake lists
`CUDA_COMMON_GPU_ARCHITECTURES = "3.5" "5.0"`. Building latest torch against
**CUDA 12.6** restores Maxwell kernels. This wheel **is the gate** for the whole
`vllm-maxwell` stack — NCCL bindings, csrc attention, and the plugin all link it.

## Key facts

- **Cross-compiles without a Maxwell GPU.** `nvcc` emits sm_50 cubins on any CUDA 12.6
  host; no device is touched at build time. → CI builds on GitHub-hosted runners.
- **CUDA must be 12.6.** This is the only hard floor. CUDA 13 removed sm_50;
  `build.sh` hard-fails on CUDA ≥ 13. (12.8 builds sm_50 too, but we standardize
  on 12.6 to match every other artifact.)
- **torch can be latest.** Not a ceiling — v2.11 source-builds sm_50 on 12.6.
  *Risk:* a newer kernel may carry an `#if __CUDA_ARCH__ >= 700` guard that fails
  to compile for Maxwell; the build proves it. Fallback: torch 2.10.
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

Output: `torch-2.11.0-cp312-cp312-linux_x86_64.whl` → published to GH Releases + Pages index.

## Validation (requires real M10 — self-hosted runner)

```python
import torch
assert torch.cuda.is_available()
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = x @ x            # exercises sm_50 FMA matmul path
torch.cuda.synchronize()
print("sm_50 matmul OK", y.sum().item())
```

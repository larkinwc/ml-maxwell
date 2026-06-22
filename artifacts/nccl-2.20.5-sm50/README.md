# nccl-2.20.5-sm50

NCCL **2.20.5** built from source for Maxwell (**sm_50**). The tensor-parallel
communication backbone for the Maxwell decode tier.

## Why this exact version

| Version | Result on Maxwell |
|---|---|
| 2.30.7 | ❌ fails at comm init — eager `cudaMemPoolCreate` (CUDA stream-ordered allocator). Maxwell: `cudaDevAttrMemoryPoolsSupported=0`. `NCCL_CUMEM_ENABLE=0` does not help. |
| **2.20.5** | ✅ **all-reduce passes on 2/4/8 GPUs.** Predates the mempool dependency. |

NVIDIA's prebuilt NCCL `.so` ships sm_60+ only (PTX sm_120) — no sm_50 SASS — so a
source build is required regardless of version.

## Transport reality

GM107 silicon **never implemented GPU P2P** (`canAccessPeer=0` for every pair,
including the on-card pair sharing one PLX bridge). NCCL therefore uses its
**host-staged SHM transport**. Measured all-reduce busBW (latency/dependency-bound,
not bandwidth-bound):

| Config | busBW (large msg) |
|---|---|
| TP=2 | 3.85 GB/s |
| TP=4 | 2.48 GB/s |
| TP=8 | 1.28 GB/s |

Negative scaling — everything funnels through host RAM (~18.8 GB/s ceiling, only
2 of 4 channels populated). **Implication:** prefer many small-TP replicas over big TP.

## Build

```bash
CUDA_HOME=/usr/local/cuda-12.6 OUTPUT_DIR=./dist ./build.sh
```

## Prebuilt artifact

The proven `.so` lives on `tyangpu1` at
`~/maxwell-stack/artifacts/nccl-2.20.5-sm50/lib/libnccl.so.2.20.5` (73 MB,
cuobjdump confirms `arch = sm_50, sm_60`). Published to GH Releases.

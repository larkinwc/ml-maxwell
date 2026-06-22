#!/usr/bin/env bash
# Build NCCL 2.20.5 from source with Maxwell (sm_50) support.
#
# WHY 2.20.5 (not latest): NCCL 2.30+ eagerly calls cudaMemPoolCreate (init.cc),
# the CUDA stream-ordered allocator (cudaMallocAsync). Maxwell reports
# cudaDevAttrMemoryPoolsSupported=0, so 2.30 dies with "operation not supported"
# at comm init. NCCL_CUMEM_ENABLE=0 does NOT fix it (separate code path).
# 2.20.5 predates that dependency (cudaMemPoolCreate appears 0x).
#
# PROVEN: all-reduce passes on 2/4/8 GPUs across both PLX clusters on tyangpu1.
# Transport is host-staged SHM (GM107 has no GPU P2P — silicon limitation).
set -euo pipefail

NCCL_VERSION="${NCCL_VERSION:-v2.20.5-1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
# Maxwell only. Add 60 too so it links cleanly against torch's default min.
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52}"

WORK="${WORK:-/tmp/nccl-build}"
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
git clone --depth 1 --branch "$NCCL_VERSION" https://github.com/NVIDIA/nccl.git
cd nccl

make -j"$(nproc)" src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$NVCC_GENCODE"

echo "[nccl-sm50] built:"
ls -la build/lib/libnccl.so.*
echo "[nccl-sm50] arch check:"
cuobjdump build/lib/libnccl.so.2.20.5 2>/dev/null | grep -i "arch =" | sort -u

OUT="${OUTPUT_DIR:-$WORK/out}"; mkdir -p "$OUT"
cp -a build/lib build/include "$OUT/"
echo "[nccl-sm50] DONE -> $OUT/"

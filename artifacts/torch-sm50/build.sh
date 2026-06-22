#!/usr/bin/env bash
# Build PyTorch 2.11.0 from source with Maxwell (sm_50) kernels.
#
# This is the project GATE: prebuilt torch wheels (2.8+) deleted Maxwell kernels
# *from binaries built against CUDA 12.8/13* — but the SOURCE tree still accepts
# sm_50 (verified: v2.11 cmake CUDA_COMMON_GPU_ARCHITECTURES includes "5.0").
# Building against CUDA 12.6 restores Maxwell kernels at latest torch.
# Cross-compiles fine WITHOUT a Maxwell GPU (only emits sm_50 cubins).
#
# HARD FLOOR: CUDA 12.6. CUDA 12.8/13 prebuilt binaries drop Maxwell; CUDA 13
# nvcc cannot emit sm_50 at all. torch/vLLM themselves can be latest.
#
# Runs on: any x86_64 Linux with CUDA 12.6 toolkit (GitHub-hosted runner OK).
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-v2.11.0}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
# Maxwell only. 5.0 = M10/GM107. 5.2 = GM20x (Titan X Maxwell etc).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-5.0;5.2}"

# --- toolchain sanity: CUDA must be 12.x (13.x cannot emit sm_50) ---
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
NVCC_VER=$("$CUDA_HOME/bin/nvcc" --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
NVCC_MAJOR="${NVCC_VER%%.*}"
if [ "$NVCC_MAJOR" -ge 13 ]; then
  echo "FATAL: CUDA $NVCC_VER cannot target sm_50 (Maxwell removed in CUDA 13). Use CUDA 12.x." >&2
  exit 1
fi
echo "[torch-sm50] CUDA $NVCC_VER, arch=$TORCH_CUDA_ARCH_LIST, torch=$TORCH_VERSION"

# --- build config: trim everything Maxwell can't / doesn't need ---
export USE_CUDA=1
export USE_CUDNN=1
export USE_CUSPARSELT=0      # not on Maxwell
export USE_FLASH_ATTENTION=0 # tensor-core only; we use legacy csrc in vllm-maxwell
export USE_MEM_EFF_ATTENTION=0
export USE_NCCL=1
export USE_SYSTEM_NCCL=1     # link our proven nccl-2.20.5-sm50
export NCCL_ROOT="${NCCL_ROOT:-/opt/maxwell-stack/artifacts/nccl-2.20.5-sm50}"
export BUILD_TEST=0
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

WORK="${WORK:-/tmp/torch-build}"
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
git clone --depth 1 --branch "$TORCH_VERSION" --recurse-submodules \
    https://github.com/pytorch/pytorch.git
cd pytorch

$PYTHON_BIN -m pip install -r requirements.txt
$PYTHON_BIN -m pip install wheel setuptools

# Produce a wheel (not an in-place install) so it lands in the artifact index.
$PYTHON_BIN setup.py bdist_wheel

echo "[torch-sm50] built wheel(s):"
ls -la dist/*.whl
mkdir -p "${OUTPUT_DIR:-$WORK/out}"
cp dist/*.whl "${OUTPUT_DIR:-$WORK/out}/"
echo "[torch-sm50] DONE -> ${OUTPUT_DIR:-$WORK/out}/"

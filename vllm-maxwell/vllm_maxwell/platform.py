"""MaxwellPlatform — vLLM out-of-tree Platform for NVIDIA Maxwell (sm_50).

Targets vLLM v0.23.x interface (vllm/platforms/interface.py). Maxwell IS a CUDA
device, so we subclass the in-tree CUDA platform base and override only what
Maxwell does differently:

  * force fp16 (no bf16/tensor cores, no fp8)
  * route attention to our legacy-csrc backend (FMA dot-product kernels)
  * route TP comms to our host-staged NCCL 2.20.5 communicator (no GPU P2P)

Everything else (memory profiling, model loading, KV spec) inherits from CUDA.

PLUGIN-FIRST: we run against UNMODIFIED upstream v0.23. If a hard cu13/sm_70+
wall is hit that can't be patched from here, we fork vLLM inside ml-maxwell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

# v0.23: the CUDA platform is CudaPlatformBase (with Nvml/NonNvml subclasses).
from vllm.platforms.cuda import CudaPlatformBase
from vllm.platforms.interface import PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.registry import AttentionSelectorConfig


class MaxwellPlatform(CudaPlatformBase):
    # Out-of-tree platform. We still behave as CUDA for is_cuda_alike() checks
    # via inheritance, but identify as OOT so vLLM treats us as a plugin.
    _enum = PlatformEnum.OOT
    device_name: str = "maxwell"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"

    # ---- capability gates -------------------------------------------------

    @classmethod
    def supported_dtypes(cls) -> list[torch.dtype]:
        # GM107: no native bf16, no fp8. fp16 + fp32 only.
        return [torch.float16, torch.float32]

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # Inherit CUDA's config handling first.
        super().check_and_update_config(vllm_config)

        model_config = vllm_config.model_config
        if model_config is not None:
            # Force fp16: bf16 weights/KV are cast on load (matches Mooncake
            # bf16->fp16 KV transfer from the MI100 prefill tier).
            if model_config.dtype == torch.bfloat16:
                model_config.dtype = torch.float16

        # Point the worker at our Maxwell worker (inherits CUDA worker for now;
        # swap to vllm_maxwell.worker.MaxwellWorker once custom hooks are needed).
        parallel_config = vllm_config.parallel_config
        if parallel_config is not None and not parallel_config.worker_cls:
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        # TODO(cudagraph): start with enforce_eager during bring-up, then enable
        # piecewise cudagraphs once MaxwellAttention is numerically validated.
        # CUDA 12.6 supports graph capture/replay on sm_50.

    # ---- backend selection ------------------------------------------------

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,  # added in v0.23
    ) -> str:
        # Always use the legacy csrc FMA paged-attention backend on Maxwell.
        return "vllm_maxwell.attention.MaxwellAttentionBackend"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # Host-staged NCCL 2.20.5 (no GPU P2P on GM107).
        return "vllm_maxwell.communicator.MaxwellCommunicator"

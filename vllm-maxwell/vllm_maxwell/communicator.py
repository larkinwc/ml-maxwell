"""MaxwellCommunicator — host-staged NCCL 2.20.5 device communicator for sm_50.

GM107 has NO GPU P2P (canAccessPeer=0 for every pair, including the on-card
pair). NCCL therefore moves tensor-parallel collectives through host RAM via its
SHM transport. This communicator just wraps the standard CUDA communicator but
binds against our source-built NCCL 2.20.5 (set via LD_LIBRARY_PATH / NCCL_ROOT)
rather than the system NCCL, because:

  * NVIDIA prebuilt NCCL ships no sm_50 SASS.
  * NCCL 2.30+ dies on Maxwell (eager cudaMemPoolCreate, unsupported).

Performance is latency/dependency-bound, not bandwidth-bound:
  TP=2 ~3.85 GB/s, TP=4 ~2.48, TP=8 ~1.28 (negative scaling via host RAM).
=> serving strategy favors many small-TP replicas over big TP.
"""

from __future__ import annotations

from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)


class MaxwellCommunicator(CudaCommunicator):
    """Standard CUDA NCCL communicator, bound to NCCL 2.20.5 sm_50 at runtime.

    For now this is a thin marker subclass: the actual NCCL selection happens via
    environment (NCCL loaded from ml-maxwell artifact). If we later need to force
    SHM tuning or disable P2P probing explicitly, override here.

    P2P is already impossible in silicon, so no disable flag is needed — NCCL
    detects canAccessPeer=0 and falls back to SHM automatically.
    """

    pass

"""MaxwellAttentionBackend — legacy csrc paged-attention for sm_50.

Maxwell has no tensor cores, so we deliberately avoid FlashAttention / Triton
tensor-core paths and use vLLM's original hand-written paged-attention kernels
(csrc/attention/attention_kernels.cu + dtype_float16.cuh). These compute the
attention dot products with CUDA-core FMA — exactly what GM107 runs natively.

STATUS: scaffold. The plan:
  1. Subclass vLLM's existing AttentionBackend interface (v0.16
     vllm/v1/attention/backends/...).
  2. In the impl, call vllm._custom_ops.paged_attention_v1/v2 (the csrc ops)
     instead of the flash/triton path.
  3. Provide an AttentionMetadataBuilder producing block tables + seq lens in
     the layout the csrc kernel expects.

The csrc kernels still ship in vLLM main (confirmed present May 2026); they only
need fp16/fp32 dtype headers, all sm_50-legal (no mma.sync / cp.async).
"""

from __future__ import annotations

# NOTE: intentionally minimal until torch-sm50 exists and we can validate the
# kernel on real M10 hardware (the make-or-break numerics test). Filling in the
# AttentionBackend/Impl/MetadataBuilder against the v0.16 interface is the next
# task once `import torch` works on the box.


class MaxwellAttentionBackend:
    """Placeholder. Will subclass vllm.v1.attention.backends.* AttentionBackend.

    Wiring target (v0.16):
      get_name()                -> "MAXWELL"
      get_impl_cls()            -> MaxwellAttentionImpl  (calls csrc paged_attention)
      get_metadata_cls()        -> MaxwellAttentionMetadata
      get_builder_cls()         -> MaxwellAttentionMetadataBuilder
      get_kv_cache_shape(...)   -> same layout as in-tree paged KV (Mooncake-compatible)
    """

    @staticmethod
    def get_name() -> str:
        return "MAXWELL"

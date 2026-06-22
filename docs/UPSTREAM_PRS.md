# Upstream PR tracker

The plugin keeps the in-tree diff near zero. The few unavoidable sm_50 touchpoints
should go **upstream**, each gated behind a capability check so it is a no-op on
modern hardware. This preserves the day-0 goal and is the "upstream-compatible with
our optimizations in place" posture.

## Candidate upstream changes (to confirm against v0.16 once torch-sm50 exists)

| Area | Likely change | Gating |
|---|---|---|
| Build (`setup.py` / CMake) | Allow `TORCH_CUDA_ARCH_LIST` to include `5.0;5.2` without hard error | only when arch explicitly requested |
| Attention selector | Let `MAXWELL`/legacy csrc backend be selectable when `cc < 7.0` | `get_device_capability()[0] < 7` |
| dtype checks | Permit fp16 path where bf16 is assumed | when bf16 unsupported by device |
| Cudagraph | Confirm piecewise capture works on sm_50 (likely no change needed) | — |

## Triton finding (informational — NOT taking this path)

Empirically, **Triton 3.5.1 DOES compile to sm_50**, including `tl.dot` (lowered to
**FMA, not MMA** — `mma=False, fma=True`, valid 150 KB cubin). So "Triton floor is
sm_70" is false at the compiler level. We still chose **legacy csrc** because its
kernels are *designed* for CUDA-core FMA and are lower-maintenance than patching
Triton capability guards. Kept here in case the csrc path hits a wall.

> Caveat: compile success ≠ verified execution. Not yet run on-device (blocked on
> torch-sm50, same gate as everything else).

## Hardware constraints that are NOT fixable (context for reviewers)

- **No GPU P2P** on GM107 — silicon never implemented peer-DMA (added in Pascal
  P100+). TP comms are host-staged. Not a bug, not configurable.
- **CUDA 13 removed sm_50.** The native floor is permanently CUDA 12.6 / torch 2.9.

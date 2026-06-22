# Fork-patch / upstream-PR tracker

Strategy: **fork vLLM** (`larkinwc/vllm`, branch `maxwell/v0.23`, embedded here as a
submodule) and keep the diff **minimal and capability-gated**, so rebasing onto each
new upstream release is mechanical and we retain day-0 model support.

Two buckets:
- **Upstreamable** patches (capability-gated, no-op on modern HW) → also send as PRs
  to `vllm-project/vllm` so our fork diff shrinks over time.
- **Fork-only** patches (Maxwell-specific, not upstreamable) → live permanently in
  the fork, kept as small as possible.

The plugin package (`vllm-maxwell`) still holds everything that *can* live
out-of-tree (Platform, Attention backend, Communicator); the fork holds only what
genuinely must touch core.

## Candidate changes (to confirm against v0.23 once torch-sm50 exists)

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
- **CUDA 13 removed sm_50.** The only hard floor is the CUDA 12.6 toolkit; torch
  and vLLM themselves can be latest (built against 12.6).

## Rebase workflow (keeping day-0)

```bash
# inside the vllm submodule (larkinwc/vllm)
git fetch upstream                 # upstream = vllm-project/vllm
git rebase v0.24.0 maxwell/v0.23   # replay our minimal diff onto the new tag
# resolve (should be small if patches stay capability-gated), retest on M10
git push origin maxwell/v0.24 -f
# then bump the submodule pointer in ml-maxwell
```

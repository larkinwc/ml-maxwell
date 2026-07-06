# Maxwell decode-tier optimization roadmap

Working plan for squeezing real performance out of the sm_50 decode tier now that
the stack is **functional end-to-end** (Qwen3.5-9B GGUF, TP=4 + CUDA graphs,
coherent output — see baseline below). Ordered by expected payoff per unit of
effort, with references to prior art so kernel work starts from proven designs.

Status: draft, 2026-07-06. Companion docs: `STATUS.md` (build/validation state),
`FP16_ON_MAXWELL.md` (fp16-emulation design), `vllm-maxwell-core/tools/maxwell/README.md`
(Qwen3.5 GGUF tooling + TP benchmarks).

---

## 0. Ground truth

Everything below is anchored to what the hardware actually is. Optimizing against
wrong assumptions on this generation is fatally easy.

### Per-die (GM107, sm_50 — 16 dies across 4× Tesla M10)

| Fact | Value | Consequence |
|---|---|---|
| fp32 cores / SMM | 128 × 5 SMM | small die; occupancy + ILP tuning matters |
| fp16 ALUs | **none** (storage/convert only) | all math fp32-accumulate; fp16 is a storage format |
| `__dp4a` int8 dot | **none** (sm_61+) | int8 dots need the llama.cpp-style guarded fallback (4× scalar int8 FMA — correct, slower); MMQ-class tile kernels stay off |
| tensor cores / bf16 / cp.async | none | SIMT fp32 kernels only |
| VRAM / bandwidth | 8 GB (~6.9 usable); spec ~88 GB/s GDDR5, **measured ceiling 73 GB/s** (`research/bw_test.py`: copy 72.0, read 73.1) | decode is weight-bandwidth-bound; quantized weights are a bandwidth optimization, not just capacity |
| shared memory | 64 KB/SMM | classic smem-tiling budgets |
| GPU P2P | **none — GM107 never implemented peer DMA** | every TP all-reduce is host-staged (NCCL SHM) |

### Host (uno-PowerEdge-C4130, 2026-07)

| Fact | Value | Consequence |
|---|---|---|
| CPU | 1× E5-2695 v4 (18c/36t), single NUMA node | all 16 dies on node0 |
| DRAM | 4× 16 GB DDR4-2133 in **A1–A4 = 1DPC on all 4 channels** (optimal) | host-staged all-reduce ceiling is fixed silicon (~68 GB/s theoretical); adding DIMMs adds capacity, not bandwidth |
| Swap | none configured | TP=16 host-RAM headroom is real; watch worker RSS |
| Storage | Intel DC SSD (377 MB/s O_DIRECT measured) | model load no longer disk-throttled (was USB-stick root before) |
| Topology | dies {0-3},{4-7},{8-11},{12-15} PIX within board, PXB across boards, all under PCIe switches (no host-bridge hop between dies) | placement affects host-staging contention — measured below |
| Driver | 580.159.03 | R580 branch; [INFERENCE] last driver family carrying Maxwell — pin alongside CUDA 12.6 |
| Quirks | `nvidia-smi` needs ~20 s CPU to enumerate 16 dies under load; one boot-time Xid 62 logged on PCI `0f:00` | don't treat slow `nvidia-smi` as a hang; watch the `0f:00` die if TP≥8 runs misbehave |

### Toolchain pins (see README.md for the full rationale)

- **CUDA 12.6 = hard floor** (CUDA 13 cannot emit sm_50). torch 2.11 + vLLM v0.23
  build from source against it; NCCL pinned 2.20.5.
- **No Triton, no inductor on sm_50.** The floor is a *PyTorch* gate, not a Triton
  one: [`has_triton()`](https://github.com/pytorch/pytorch/blob/main/torch/utils/_triton.py)
  returns False below `major >= 7`, and inductor has no non-Triton CUDA backend
  (`cuda_backend ∈ {triton, halide, pallas}`; `cpp` is CPU-only). Triton's compiler
  itself would route sm_50 to MMA-v1 software-FMA — no *documented* silent
  miscompiles, but our own FLA/Triton GDN kernel computed ~10× wrong values on
  Maxwell, so the project rule stays: **performance kernels are hand CUDA;
  `torch.compile` stays off** (`CompilationMode.NONE`).
- **Profiling on sm_50 / CUDA 12.6 (all source-verified):** `torch.profiler`
  kernel timelines WORK (CUPTI Activity API has no Maxwell gate in 12.x — only PM
  sampling/SASS metrics need sm_75+); legacy `nvprof` still ships in 12.6
  (deprecated 12.8, removed 13.0); `cudaEvent` + in-kernel `clock64()` for
  sub-kernel timing; Nsight Systems 2024.4 (12.6 bundle) likely works — smoke-test;
  **Nsight Compute never supported Maxwell**.

### Measured baseline (2026-07-05, SSD, Qwen3.5-9B Q4_K_M GGUF, greedy)

All cells CUDA-graph mode (`FULL` → auto `FULL_DECODE_ONLY`), 8 prompts × 128
decode tokens, F16INPROJ GGUF unless noted:

| Config | batch-8 tok/s | single-stream tok/s | load | coherent |
|---|---|---|---|---|
| TP=2 | 10.3 *(prior session)* | — | — | ✓ |
| **TP=4 packed 0-3** | **18.5** | 4.4 | 149 s | ✓ |
| TP=4 spread 0,4,8,12 | 18.6 | 4.5 | 148 s | ✓ |
| TP=8 (0-7) | 14.5 | 2.2 | 1366 s | ✓ |
| **TP=16 (all dies)** | **23.5** ← new best | 3.9 | 869 s | ✓ |
| TP=4, plain Q4_K_M in_proj | 17.8 | 3.9 | 258 s | **✗ gibberish** |

Old-vs-new (the SSD A/B): TP=4-CG 18.5 → 18.5, TP=8-CG 14.2 → 14.5 — decode
throughput is storage-independent, as expected for a GPU/comm-bound loop. Eager
cells were deliberately not re-run (prior values stand: 8.3 / 12.1 / 10.7 / 6.0
for TP=2/4/8/16; eager is a comparison mode only — see the journal's warning).

**TP=16 + graphs (never run before) is the new batch-8 champion at 23.5 tok/s**
— eager's 6.0 had written it off, echoing the journal's eager-artifact lesson.
Two honest caveats: (1) **TP=8 is a valley** (18.5 → 14.5 → 23.5 is
non-monotonic) — unexplained; profile before trusting any scaling story
(Tier 0.1); (2) per-die efficiency still favors TP=4 (4.6 vs 1.5 tok/s/die at
batch 8), so for *aggregate* serving, 4× TP=4 DP replicas (~74 tok/s across 32
concurrent requests) still beat one TP=16 engine (Tier 2.3). TP=16's crown is
for a single shared engine at modest concurrency.

Single-stream vs batch-8 gap (4.4 vs 18.5 tok/s at TP=4-CG) says decode is
dominated by per-token fixed costs (kernel launches, dequant amplification,
host-staged all-reduce latency) that amortize across a batch — exactly the
costs Tiers 1–2 attack.

**Negative results — measured, do not re-run** (tyangpu1 journal + appenders):
- dp4a register-only emulation tweak: **0%** (quant ALU fully overlaps weight
  loads on GM10x — ALU micro-optimization is a dead end; it's a bandwidth game).
- NCCL algo/thread knobs & NUMA pinning: ±1% noise; `NCCL_ALGO=Tree` −18%.
- `GGML_CUDA_MMV_Y` occupancy sweep: +2% at 2, plateau after.
- SD-card→SSD storage swap: 0% on decode throughput (2026-07-05 A/B, this doc).
- TP=4 GPU placement, packed vs spread across boards: 0% batch and single-stream
  (18.5/4.4 vs 18.6/4.5) — with graphs pipelining the host-staged allreduce,
  topology is a non-lever at TP=4 on the C4130 switch fabric.

---

## 1. Tier 0 — Measure first

Cheap, do before writing any kernel:

1. **Per-kernel decode profile.** Primary: `torch.profiler` (CUPTI activity
   timeline → Chrome trace), corroborated by `nvprof --print-gpu-summary` and
   `cudaEvent` brackets for suspect kernels. Deliverable: a ranked table (dequant
   kernels / cuBLAS GEMV / GDN scan / all-reduce / launch gaps) for TP=2 and TP=4
   decode steps. `research/profile_decode.py` and `research/ncu_decode.py` are
   prior art; `research/AGENTS.maxwell.md` is the measured-results journal from
   the tyangpu1 bring-up — its hardest-won methodology lesson: **never benchmark
   eager-mode and draw TP conclusions** (the retracted "TP=4 regresses" table was
   pure eager-launch artifact; graphs flipped it to monotonic scaling).
2. **Roofline per die.** Q4_K_M weights ≈ 6.27 GB ⇒ single-stream ceiling if
   weights were read once at quantized width: `73 GB/s ÷ (6.27/TP GB)` ≈ 47 tok/s
   at TP=4. We measure 4.4 — an order of magnitude of recoverable headroom
   between dequant traffic amplification, launch overhead, and comm latency.
   Priors from tyangpu1 (Q4_0, MMVQ path): MMVQ = **62% of decode self-CUDA
   time**; isolated MMVQ streams 43 GB/s = 59% of ceiling; Q6_K real-op is the
   worst recorded kernel at **33 GB/s / 46%** — and Q4_K_M models are mostly
   Q4_K/Q6_K, so the K-quant kernels have even more layout headroom than Q4_0
   did. (Note: Nsight Compute empirically rejects sm_50 — these all came from
   torch-profiler self-times + hand microbenchmarks; same recipe applies here.)
3. **Host BW + NCCL μbench.** STREAM triad on the host; all-reduce latency/bw for
   8 KB–8 MB messages at TP=2/4/8/16 (`research/run_nccl_matrix.sh` exists).
4. **Placement A/B — DONE 2026-07-05, null effect.** Packed `0,1,2,3` vs spread
   `0,4,8,12` at TP=4+graphs: 18.5/4.4 vs 18.6/4.5 tok/s (batch/single), load
   identical. Keep the boring board-local default; spend the effort on Tier 1.
5. **llama.cpp reference number.** llama.cpp master supports Qwen3.5/GDN natively
   (`LLM_ARCH_QWEN35` in `src/llama-arch.h`) and its sm_50 build already exists
   (see `STATUS.md`). Run the same GGUF single-die and note tok/s — it exercises
   fused MMVQ + native GDN CUDA, i.e. roughly where Tier 1 wants to land vLLM.
6. **Q4_0 cross-check (mechanism probe).** The tyangpu1 journal
   (`research/AGENTS.maxwell.md`) measured **29.4 tok/s single-stream at TP=8+graphs
   on a Q4_0 GGUF through vLLM's quantized-kernel path** — ~6× today's C4130
   single-stream via dequant+cuBLAS. Different box/quant/seq-len, so indicative
   not controlled — but it strongly suggests Q4_0's vecdot avoided the broken
   `__dp4a` path that K-quants hit. Reproduce a Q4_0 build of Qwen3.5-9B on the
   C4130 and profile which kernel actually runs; if confirmed, Tier 1.1's payoff
   estimate gets our own prior measurement behind it.
7. **`max_num_seqs` concurrency sweep** (journal's open follow-up): on the best-TP
   engine, sweep 1→4→8→16 and record aggregate + per-request tok/s to find the
   batching knee. Decode is BW-bound; the knee defines the serving config.

## 2. Tier 1 — Decode kernels (biggest wins)

1. **Fused quantized matvec: sync vendored GGUF kernels from llama.cpp master
   (MMVQ path).** Root-cause insight from the bring-up: vLLM's vendored GGUF
   kernels call `__dp4a` *unguarded* → NaN on sm_50, which forced us onto
   dequant-to-fp16 + cuBLAS (read Q4 0.55 B/param + write fp16 2 B + re-read fp16
   2 B ≈ 8× the traffic of a fused kernel). Modern llama.cpp guards it:
   [`ggml_cuda_dp4a`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh)
   (`common.cuh:694`) falls back below cc 610 to 4× scalar int8 mul-adds —
   correct, and still memory-bound at decode. On sm_50, batch≤8 quantized matvec
   runs `mul_mat_vec_q` (`mmvq.cu`, 4 warps × 32, fused dequant+dot via
   `vec_dot_q4_K_q8_1`/`q6_K`, fp32 accumulate, shfl reduce) — MMQ is
   arch-disabled below dp4a and stays out. Action: refresh the vendored kernels
   (mmvq.cu + vecdotq.cuh + common.cuh ladder) to master's arch-guarded versions,
   make `MAXWELL_GGUF_DEQUANT` the fallback instead of the default, A/B against
   dequant+cuBLAS. DMMV is gone upstream (removed in
   [PR #10318](https://github.com/ggml-org/llama.cpp/pull/10318)) — MMVQ+fallback
   *is* the modern pre-dp4a path. Payoff is grounded in our own history: the
   tyangpu1 journal measured **29.4 tok/s single-stream (TP=8+graphs, Q4_0)**
   through the quantized-kernel path vs **4.4 tok/s single-stream (TP=4+graphs)**
   through today's dequant+cuBLAS path — different box/quant/seq-len (indicative,
   not controlled; see Tier 0.6), but the gap is a chasm, not a margin.
   And the fix is surgical, not architectural: the tyangpu1 stack already ran
   MMVQ with software-dp4a *everywhere quant runs* — the C4130 NaN bit the
   K-quant/GEMM path specifically, and the `MAXWELL_GGUF_DEQUANT` override then
   threw the whole fused family away. Restore it type-by-type with llama.cpp
   master's guards, keep dequant+cuBLAS for prefill/large batch (that's exactly
   llama.cpp's own dispatch shape).
   Then **re-apply the shipped SoA layout work on top, extended to K-quants**:
   `research/patch_mmvq_soa.py` + `patch_gguf_py_soa.py` repacked Q4_0's 18-byte
   AoS blocks into 16B-aligned SoA (+19% kernel BW, 48.7→58.3 GB/s; e2e **+9.5%
   TP=1 / +5.5% TP=4 / +2.7% TP=8**; correctness max_abs ≈1e-5), with
   `VLLM_GGUF_DECODE_ONLY=1` freeing the AoS copy (+18% KV capacity, fits 9B at
   TP=1). Q6_K's 210-byte superblocks (33 GB/s / 46% measured) should benefit
   even more than Q4_0 did.
2. **Port llama.cpp's GDN delta-rule kernel.** llama.cpp master ships
   [`gated_delta_net.cu`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated_delta_net.cu)
   (~330 lines): pure fp32, **zero** `__CUDA_ARCH__`/fp16/dp4a gates, shfl-only
   reductions, warp-per-state-column, S_v ∈ {16,32,64,128}, plus a fused-cache
   variant. It is the exact Qwen3.5 recurrence our fork runs torch-native today —
   the cleanest, lowest-risk port with the best launch-count payoff (the
   torch-native scan is dozens of eager ops per layer per step).
3. **Prefill/full-attention reference: llama.cpp FA runs on sm_50.** No arch
   floor on `FLASH_ATTN_AVAILABLE`; decode picks `fattn-vec`, prefill
   `fattn-tile`, both with complete fp32 branches (no `FP16_AVAILABLE` below
   Pascal; `ggml_cuda_mad(float&, half2, half2)` converts and FMAs in fp32).
   Use as the porting source if profiling shows our prefill attention path is a
   bottleneck on the few full-attention layers. Skip MMA/WMMA FA and MMQ tile
   kernels entirely — hard-gated on hardware sm_50 lacks.
4. **Launch-count reduction around the block** (RMSNorm/residual/rope fusions):
   graphs already hide launch cost in steady-state decode; fusion still shrinks
   graph capture time, prefill, and non-captured paths.
5. **Kernel tuning discipline for GM107** (source-verified refs; all fp32-SIMT era):
   - **maxas SGEMM** ([repo](https://github.com/NervanaSystems/maxas),
     [sgemm walkthrough](https://github.com/NervanaSystems/maxas/wiki/sgemm)) — ~98% of
     Maxwell peak; the wiki calls the **64-thread variant ideal for GM107**. Port to
     CUDA C: 8×8 register blocking under 128 regs/thread, vec4/128-bit weight loads
     (`__ldg`/`const float4* __restrict__` → read-only cache), double-buffered smem
     with XOR swap + zigzag swizzle (4-byte banks have undocumented conflict
     patterns), swirl FFMA ordering. Validate with `cuobjdump -sass` (no spills,
     `LDG.E.128` emitted).
   - **NervanaGPU “pseudo fp16”** ([repo](https://github.com/VisionSystemsInc/nervanagpu))
     — fp16-storage/fp32-compute GEMM is the architectural template; extend one level
     down to int4-storage. Bonus: its source proves cuBLAS fp16 GEMM required
     **cc ≥ 5.2** (on 5.0 it upconverted to fp32) — a custom kernel is mandatory, not
     optional, on the M10.
   - **CUTLASS SIMT Sm50** ([simt_sgemm_nn_sm50.cu](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/simt_sgemm_nn_sm50.cu),
     [mma_sm50.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm50.h))
     — still compiles (sm_50 in default arch list) but officially unsupported since
     3.0 (floor = SM70). Useful as a tiled-GEMM baseline + ~30-shape tile sweep for
     prefill; the M=1 decode GEMV must be hand-written (warp-per-row + `__shfl` fp32
     reduce). No int4 atoms below sm_75 — dequant lives in our K-loop.
   - **Hard budgets** ([Maxwell tuning guide](https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html)):
     ≤32 KB smem/block (keep ≥2 blocks/SMM of the 64 KB), ≤255 regs/thread, 32
     blocks/SMM max, 4-byte smem banks, global loads are L2-only unless `__ldg`.
   - **maxDNN** ([arXiv:1501.06633](https://arxiv.org/abs/1501.06633)) — 96% efficiency
     conv on the maxas tile; its “lift index math into constant memory” trick applies
     to Q4_K block/group offset arithmetic.

## 3. Tier 2 — Comms & scheduling

1. **Interconnect tuning is (measured) a dead end — don't re-litigate it.** The
   tyangpu1 journal ran the matrix at TP=8+graphs: NUMA pinning, `NCCL_ALGO=Ring`,
   `NCCL_NTHREADS`, `NCCL_P2P_DISABLE` — all within ±1%; **`NCCL_ALGO=Tree` was
   −18%, never force it**. With CUDA graphs capturing the whole decode step
   (collectives included), the host-staged allreduce pipelines behind compute and
   stops being the bottleneck. The lever is per-die work (Tier 1), not comms knobs.
   The C4130 packed-vs-spread A/B (Tier 0.4) confirmed it: even placement is a
   null effect at TP=4 with graphs on (18.5/4.4 vs 18.6/4.5). Keep board-local
   groups as the boring default and move on.
2. **Async scheduling — enable and verify (HIGH, low effort).** Pure CPU-side,
   no arch gate: `MultiprocExecutor.supports_async_scheduling() = True`, and it
   auto-enables unless pooling/non-EAGLE-spec-decode. Overlaps scheduling with GPU
   execution — exactly what a launch-bound target wants on top of FULL cudagraphs.
   Verify it actually engages for a hybrid GDN model (nothing blocks it in
   `vllm/config/vllm.py`'s checks, but that's [INFERENCE], not smoke-tested).
   Note: v0-style multi-step scheduling no longer exists standalone in v1 — the
   launch-overhead levers are cudagraphs + async sched, period.
3. **Data-parallel replicas beat big TP on this silicon.** The fork's own
   `MAXWELL.md` says it: *prefer many small-TP replicas over large TP* — no P2P
   means TP>4 buys mostly all-reduce pain. 16 dies = 4 independent TP=4 engines
   behind a router ≈ 4× the TP=4 throughput for aggregate serving, zero kernel
   work. Worth standing up before any deeper comms tuning.

## 4. Tier 3 — Feature backports

Verdicts from source inspection of the fork + upstream issue tracker (2026-07):

| Feature | sm_50 verdict | Why / evidence |
|---|---|---|
| Async scheduling | ✅ do it (Tier 2.2) | CPU-side; no arch gate |
| ngram spec decode | ⛔ **blocked by GDN rollback** | proposer is CPU numba (fine), but hybrid state rollback corrupts output upstream ([#39273](https://github.com/vllm-project/vllm/issues/39273)) and the rollback kernel (`postprocess_mamba_fused_kernel`) is Triton |
| EAGLE / Qwen3.5 MTP | ⛔ same blocker + `align` cache mode required | `Qwen3NextMTP` raises outside align mode; rollback path identical |
| Prefix caching (hybrid) | 🟡 later | align-mode only; Triton `do_mamba_copy_block` state copy (no torch fallback); 0% hit below block granularity ([#40696](https://github.com/vllm-project/vllm/issues/40696)); low value for a decode tier |
| Chunked prefill | 🟡 usable with align constraints | scheduler forces block-aligned splits; verify kernel selection (below) |
| KV cache fp8 | ⛔ avoid | bf16→fp8 convert is `assert(false)`-stubbed below sm_80 → garbage under `-DNDEBUG`; fp16→fp8 works but is software converts on a BW-bound part |
| KV cache int8 | 🟡 explore | listed dtype; arch-cleanliness on sm_50 [UNVERIFIED]; hybrid GDN keeps KV small anyway (only ~¼ of layers are full attention) |
| FlashAttention / FlashInfer | ⛔ hard-gated | FA needs cc≥8.0, FlashInfer 7.5–12.1; fork's C++ `paged_attention` (3e-4 max err on M10) is the proven floor |

**Spec-decode prerequisite (the one worth doing):** a Triton-free GDN state
rollback — either pure-torch checkpoint/restore mirroring the fork's
`_torch_chunk_gated_delta_rule` approach, or the ReplaySSM input ring-buffer idea
([#46187](https://github.com/vllm-project/vllm/issues/46187)). Once rollback is
trustworthy, CPU ngram is the first variant to light up (zero GPU requirements),
and MTP/EAGLE become candidates after.

**Two routing verifications for Tier 0** (both cheap, both about Triton kernels
silently in the path):
- Confirm which GDN *decode-step* kernel actually runs on sm_50
  (`fused_recurrent_gated_delta_rule_packed_decode` is FLA/Triton upstream;
  output is coherent, so whatever runs is working — pin it down and gate it).
- Confirm the chunked-prefill/paged-decode *Triton* kernel is not selected for
  Qwen3.5's non-power-of-2 attention block size (144) — the C++ paged_attention
  path must win.

Plus, from the Qwen3.5 bring-up follow-ups:
- **Plain-quantized pipeline** (drop `--f16-inproj`): **RETESTED 2026-07-05 —
  still broken, keep the F16 rewrite.** `Qwen3.5-9B-FIXED.gguf` (plain Q4_K_M
  in_proj) runs at full speed (17.8 tok/s batch-8, TP=4+graphs) but emits
  gibberish (`",akai方-的… erce :erce"`), while the same-day F16INPROJ control
  is coherent at 18.5. So the `__dp4a` dequant fix did NOT cover the merged
  quantized `in_proj_qkvz` loader path — suspect the split slices Q4_K
  superblocks mid-block. Fix candidates: superblock-aligned GGUF split in
  `linear.py`, or dequantize only `in_proj` tensors at load (a targeted,
  in-memory version of what `--f16-inproj` does on disk).
- Generalize `fix_gguf.py` GDN dims beyond 9B.
- **TP≥8 load time**: 1366 s at TP=8, 869 s at TP=16 (vs 149 s at TP=4) —
  CPU-side per-worker GGUF processing, not disk (SSD idles). Now that TP=16 is
  the batch-throughput champion, pre-sharding/caching per-rank weights is worth
  real effort (14× load-time gap vs TP=4 hurts every restart).

## 5. Tier 4 — Hygiene & upstreaming

- Drop the `qwen3_next.py` debug-probes commit before upstreaming; gate the GDN
  probes out of `qwen_gdn_linear_attn.py`.
- Keep the fork minimal-diff + capability-gated (per README) so rebases onto new
  upstream vLLM releases stay mechanical.
- Track fork-vs-upstream deltas in `docs/UPSTREAM_PRS.md`.

---

## Appendix A — TP retest on SSD (2026-07-05)

Motivation: prior TP results were collected while the root filesystem lived on a
USB stick; after moving to an Intel DC SSD (`SSDSC1BG400G4R`, 377 MB/s O_DIRECT
measured on the model file) we re-ran a control/treatment canary to test whether
storage had depressed the high-TP numbers.

**Verdict: no.** Decode throughput is unchanged (control TP=4-CG 18.5 → 18.5
exact; treatment TP=8-CG 14.2 → 14.5, within noise), so the remaining eager
cells were not re-run. What storage *also* did not fix: TP=8 load time (1366 s)
— the load path is CPU-side per-worker GGUF processing, not I/O.

Method: `tp_bench.py` (RESULT_JSON), one fresh process per config, driven by
`~/bench_ssd/run_matrix2.sh` on `uno-poweredge-c4130` with per-run zombie
cleanup, `vmstat` sampling, and `free`/env snapshots. Raw logs:
`~/bench_ssd/stageA/` (canary + plain-Q4_K_M probe) and `~/bench_ssd/stageB/`
(placement A/B + TP=16) on the box. Box state during runs: no swap configured,
62 GB RAM, driver 580.159.03, CUDA 12.6, branch `maxwell/qwen35-gguf-sm50`
(core @ `c64508cc4` — adds the `BENCH_MODEL` override used for the plain-quant
probe).

Full result rows (RESULT_JSON, abridged):

| run | tp | cg | batch tok/s | single tok/s | load_s | coherence |
|---|---|---|---|---|---|---|
| stageA/tp4-cg | 4 | ✓ | 18.5 | 4.4 | 149.1 | "Paris." ✓ |
| stageA/tp8-cg | 8 | ✓ | 14.5 | 2.2 | 1365.5 | "Paris." ✓ |
| stageA/tp4-cg-q4km (FIXED.gguf) | 4 | ✓ | 17.8 | 3.9 | 258.1 | gibberish ✗ |
| stageB/tp4-cg-spread (0,4,8,12) | 4 | ✓ | 18.6 | 4.5 | 148.1 | "Paris." ✓ |
| stageB/tp16-cg | 16 | ✓ | **23.5** | 3.9 | 869.3 | "Paris." ✓ |

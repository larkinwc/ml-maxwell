# Maxwell (Tesla M10 / sm_50) Inference — Agent Notes

Operational findings for running vLLM (`vllm-maxwell-core`, branch `maxwell/v0.23`)
on 2× Tesla M10 (8 GM107 dies, sm_50). Decode tier of the prefill(MI100)→decode(M10)
split. **The metric is output tok/s.**

## Tensor-parallel scaling (MEASURED — TWO corrections to earlier claims)

> Correction 1: "TP doesn't help on M10 because GM107 has no P2P." **Wrong** — TP scales.
> Correction 2 (BIGGER): our first scaling table ran with `enforce_eager=True`, which
> disables CUDA graphs. That eager overhead — NOT the interconnect — was the dominant
> cost and it hit higher-TP configs hardest, producing a fake "TP=4 regresses" result.
> **With CUDA graphs ON, TP=4 is the throughput winner and scaling is monotonic.**

Measured, Qwen3.5-9B Q4_0, single-stream greedy decode, 128 tokens, `max_model_len=320`,
post-warmup (Triton JIT + graph capture excluded), **switch-local dies**:

| Config | EAGER (old, misleading) | **GRAPHS ON (correct)** | graph speedup | vs TP=1 |
|--------|-------------------------|--------------------------|---------------|---------|
| TP=1   | 6.84                    | **7.56**                 | +11%          | 1.00×   |
| TP=2   | 12.09                   | **14.02**                | +16%          | 1.85×   |
| TP=4   | 11.19 (fake regression) | **23.15**                | **+107%**     | 3.06×   |
| TP=8   | 10.50                   | **29.42**                | **+180%**     | **3.89×** |

**DEPLOYMENT GOAL: maximize total decode throughput, then take whatever concurrency
comes with it.** Not minimizing single-stream latency. Under that goal:

- **Default to TP=8 (all 8 M10 dies as one engine).** It is the throughput winner at
  29 tok/s single-stream and scales monotonically (3.89× over TP=1). Concurrency is then
  layered on top by raising `max_num_seqs` (continuous batching) on that one TP=8 engine —
  decode is memory-BW-bound, so batching more sequences adds aggregate tok/s at little
  marginal cost until HBM/compute saturates.
- Lower-TP configs (TP=2/4) are only for: fitting *multiple different models* at once, or
  carving the box into independent replicas. For a single 9B decode service, TP=8 wins.

Takeaways:
- **NEVER set `enforce_eager=True` for decode.** It was never needed and nearly triples
  TP=8 throughput (and doubles TP=4). CUDA graphs work on sm_50.
- With graphs, scaling is **monotonic all the way to TP=8** (best single-stream = 29 tok/s).
  The earlier "TP=4/8 regress" / "TP=2 is the sweet spot" claims were purely eager-mode
  artifacts and are RETRACTED.
- **Why eager hit TP hardest:** in eager mode every kernel launches from Python per token
  (N× launch overhead at TP=N) and each per-layer host-staged NCCL allreduce forces a
  CPU↔GPU sync that can't overlap compute. CUDA graphs capture the whole decode step
  (all layers + collectives) into one replay, collapsing launch overhead and pipelining
  the allreduces. More ranks ⇒ more overhead to amortize ⇒ bigger graph win (TP=4 +107%
  vs TP=1 +11%).
- **Open follow-up (to push aggregate even higher):** sweep `max_num_seqs` (e.g. 1→4→8→16)
  on the TP=8 engine and record aggregate tok/s + per-request tok/s to find the
  concurrency knee. Expectation: aggregate keeps climbing past single-stream 29 tok/s
  until per-die HBM bandwidth saturates.

### NCCL / NUMA tuning gave NOTHING (TP=8, graphs on, single-stream)
Tested matrix (Qwen3.5-9B Q4_0, TP=8, all knobs vs baseline 29.58 tok/s):

| Config | tok/s | verdict |
|--------|-------|---------|
| baseline | 29.58 | — |
| `numactl --cpunodebind=1 --membind=1` | 29.31 | noise |
| `NCCL_P2P_DISABLE=1 NCCL_ALGO=Ring` | 29.48 | noise |
| `NCCL_ALGO=Tree` | 24.16 | **worse (-18%), avoid** |
| numa + Ring | 29.31 | noise |
| numa + Ring + `NCCL_NTHREADS=256` | 29.39 | noise |

- **NUMA pinning and NCCL algo/thread knobs do not help.** All within ±1% of baseline
  except `NCCL_ALGO=Tree`, which actively hurts — leave NCCL on its default (Ring-ish)
  auto choice and do NOT force Tree.
- **What this means:** with CUDA graphs on, the host-staged allreduce is already pipelined
  and is **no longer the bottleneck at TP=8**. The remaining limiter is per-die GDN decode
  compute (memory-BW-bound on GM107, ~72 GB/s HBM). To go faster you must reduce per-die
  work (better quant, larger batch to amortize, faster kernels), not tune the interconnect.

### CUDA graphs on the hybrid GDN model
- sm_50 supports CUDA graphs; they capture fine.
- `GDNAttentionBackend` only advertises `AttentionCGSupport.UNIFORM_BATCH`, so vLLM cannot
  use `CUDAGraphMode.FULL` and **auto-downgrades to `FULL_DECODE_ONLY`** (logged as a
  WARNING — this is expected/benign). Prefill stays eager, decode is graphed — exactly the
  regime that matters here.

### PCIe switch topology — only matters for PARTIAL TP groups
`nvidia-smi topo -m`: dies **0-3 share one PCIe switch (PIX)**, **4-7 share another (PIX)**,
and **0-3 ↔ 4-7 is PXB** (slower, crosses host bridges). `nvidia-smi topo -p2p r` shows
**`GNS` (P2P not supported)** between all dies → every allreduce is host-staged.
- **TP=8 spans both switches by definition and is still the throughput winner** — once
  CUDA graphs pipeline the allreduce, the cross-switch PXB hop is not the bottleneck
  (see NCCL/NUMA matrix: tuning it changes nothing). So for the default TP=8 service,
  topology is a non-issue; just use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
- **Switch-locality only matters when you run a PARTIAL TP group** (TP=2/4): keep it within
  one switch — `{0,1,2,3}` or `{4,5,6,7}` — never straddle (e.g. 1,2,3,4). Our original
  TP=4 used 1,2,3,4 and paid the PXB tax (one more reason that early table was misleading).
- All dies are on **NUMA node 1** (CPUs `8-15,24-31`), but NUMA pinning made no measurable
  difference (matrix above), so it is optional.

## TP is NOT capped by KV-head count (GQA heads are replicated)

> Earlier (wrong) claim here was "max usable TP = number of KV heads, so 9B (4 KV heads)
> can't do TP=8." **False.** `num_kv_heads=4` is the real GQA architecture (identical in
> AWQ and GGUF) and does NOT cap TP. When `tp_size > total_num_kv_heads`, vLLM replicates
> KV heads across ranks (`num_kv_heads=1`, `num_kv_head_replicas=tp_size/total_num_kv_heads`).
> AWQ already did this and ran 9B at TP=8 fine.

- The GGUF QKV weight loader had a bug: it sharded q/k/v naively by `size // tp_size`,
  ignoring `num_kv_head_replicas`, so each rank got a *fraction* of a KV head. Downstream
  `qkv.split([...])` in `qwen3_next.py` then crashed
  (`split_with_sizes ... split_sizes=[1024,256,256]` vs tensor 1280 at TP=8).
- Fixed in commit `<QKV-KV-REPLICA>`: size k/v shards by KV-head count and offset by
  `tp_rank // num_kv_head_replicas`, mirroring the non-GGUF `load_qkv_weight` path.
- After the fix, **all of TP ∈ {1,2,4,8} run and produce coherent output for 9B GGUF.**

## GPU die map (IMPORTANT)

- `nvidia-smi` shows 12 dies (0-11), ALL reported as "Tesla M10".
- **Dies 0-7 are the 2× Tesla M10 (sm_50) — use these.**
- **Dies 8-11 are a DIFFERENT GPU — do NOT use for the M10 decode tier.**
- Always keep `CUDA_VISIBLE_DEVICES` within `0..7`. Default TP=8 service uses all of them:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.

## Running benchmarks / inference (recap)

- Background to a log + poll; SSH drops exit 255 on long foreground commands.
- **Do NOT pass `enforce_eager=True`** (default eager-off keeps CUDA graphs; see scaling).
- TP≥2 needs `VLLM_WORKER_MULTIPROC_METHOD=spawn`; always `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Must pass `tokenizer="Qwen/Qwen3.5-Xb"` (original HF tokenizer); GGUF tokenizer fails
  the multimodal check.
- After a killed TP run, the process often **hangs in NCCL/multiproc teardown AFTER
  printing the result** and may leave an orphaned `VLLM::EngineCore` holding ~7 GB on a
  die. Verify with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  and `kill -9 <pid>` the orphan before reusing that die.
- `pkill`/`kill` over SSH frequently returns exit 255 even when it worked — verify with
  `pgrep`/`nvidia-smi`, don't trust the exit code.

## Correctness fixes that gate the above (committed on `maxwell/v0.23`)

- `eba85f5d8` — Qwen3.5 GGUF GDN V-head reorder (9B `num_v_heads != num_k_heads`); without
  it 9B emits gibberish. No-op when `num_v == num_k` (0.8B).
- `d121673bc` — GGUF MergedColumnParallel: narrow pre-fused qkv tuple shards per sub-block;
  fixes TP≥2 gibberish (was slicing the fused q+k+v tensor contiguously across ranks).

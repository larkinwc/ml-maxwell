#!/bin/bash
# NCCL / NUMA tuning matrix for TP=8 decode on M10 (host-staged allreduce).
# Each run: fresh process, graphs ON, switch group = all 8 M10 dies.
cd /home/larkinwc/maxwell-stack/vllm-maxwell-core
PY=/home/larkinwc/miniconda3/envs/vllm-maxwell/bin/python
BENCH=/home/larkinwc/maxwell-stack/bench_9b_prof.py
COMMON="BENCH_TP=8 BENCH_EAGER=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
LOG=/home/larkinwc/maxwell-stack/nccl_matrix.log
: > $LOG

run() {
  local name="$1"; shift
  local extra="$1"; shift
  local wrap="$1"; shift
  echo "=================== RUN: $name ===================" >> $LOG
  echo "ENV: $extra   WRAP: $wrap" >> $LOG
  eval "env $COMMON $extra $wrap $PY $BENCH" >> $LOG 2>&1
  # cleanup any lingering procs between runs
  pkill -9 -f bench_9b_prof.py 2>/dev/null
  sleep 5
}

# 1. baseline (no extra knobs) -- reproduces the 29.42 number for this driver
run "baseline" "" ""
# 2. NUMA pin to node 1 (cpu+mem)
run "numa_node1" "" "numactl --cpunodebind=1 --membind=1"
# 3. NCCL explicit P2P disable + Ring algo
run "nccl_ring" "NCCL_P2P_DISABLE=1 NCCL_ALGO=Ring" ""
# 4. NCCL Tree algo
run "nccl_tree" "NCCL_P2P_DISABLE=1 NCCL_ALGO=Tree" ""
# 5. NUMA pin + Ring
run "numa_ring" "NCCL_P2P_DISABLE=1 NCCL_ALGO=Ring" "numactl --cpunodebind=1 --membind=1"
# 6. NUMA pin + Ring + more NCCL threads
run "numa_ring_nthreads" "NCCL_P2P_DISABLE=1 NCCL_ALGO=Ring NCCL_NTHREADS=256" "numactl --cpunodebind=1 --membind=1"

echo "=================== MATRIX DONE ===================" >> $LOG
grep -E "RUN:|RESULT" $LOG >> $LOG

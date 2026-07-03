#!/bin/bash
# Sweep GGML_CUDA_MMV_Y to test occupancy effect on the memory-bound MMVQ decode kernel.
# Edits the #define, does an incremental rebuild (only gguf object recompiles), benches TP=1.
set -u
CORE=/home/larkinwc/maxwell-stack/vllm-maxwell-core
HDR=$CORE/csrc/libtorch_stable/quantization/gguf/ggml-common.h
PY=/home/larkinwc/miniconda3/envs/vllm-maxwell/bin/python
LOG=/home/larkinwc/maxwell-stack/sweep_mmvy.log
: > $LOG

for Y in 2 4 8; do
  echo "=================== MMV_Y=$Y ===================" | tee -a $LOG
  # set the define
  sed -i -E "s/#define GGML_CUDA_MMV_Y [0-9]+/#define GGML_CUDA_MMV_Y $Y/" $HDR
  grep "define GGML_CUDA_MMV_Y" $HDR | tee -a $LOG
  # incremental rebuild
  cd $CORE
  echo "[build start $(date +%T)]" >> $LOG
  bash /home/larkinwc/maxwell-stack/build_vllm.sh >> /home/larkinwc/maxwell-stack/sweep_build_$Y.log 2>&1
  echo "build_exit=$(grep -o 'VLLM_BUILD_EXIT=[0-9]*' /home/larkinwc/maxwell-stack/sweep_build_$Y.log | tail -1)" | tee -a $LOG
  # bench TP=1 graphs-on
  env CUDA_VISIBLE_DEVICES=0 BENCH_TP=1 BENCH_EAGER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
      $PY /home/larkinwc/maxwell-stack/bench_9b_prof.py >> $LOG 2>&1
  pkill -9 -f bench_9b_prof.py 2>/dev/null
  sleep 5
done
echo "=================== SWEEP DONE ===================" | tee -a $LOG
grep -E "MMV_Y=|RESULT|build_exit" $LOG | tee -a $LOG

import os, glob, json
from collections import defaultdict
from vllm import LLM, SamplingParams
p="/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_0.gguf"
TRACE=os.environ["VLLM_TORCH_PROFILER_DIR"]
TP=int(os.environ.get("BENCH_TP","1"))
def main():
    llm=LLM(model=p, tokenizer="***************", trust_remote_code=True, dtype="float16",
            max_model_len=320, max_num_seqs=1, gpu_memory_utilization=0.90,
            enforce_eager=True, tensor_parallel_size=TP,  # eager so per-op kernels show in trace
            profiler_config={"profiler":"torch","torch_profiler_dir":TRACE,
                             "torch_profiler_record_shapes":True})
    # warmup
    llm.generate(["warmup pass one two three"], SamplingParams(temperature=0.0,max_tokens=16))
    # profile a clean decode of ~64 tokens
    llm.start_profile()
    llm.generate(["The capital of France is"], SamplingParams(temperature=0.0,max_tokens=64))
    llm.stop_profile()
    print("PROFILE_DONE")
if __name__=="__main__": main()

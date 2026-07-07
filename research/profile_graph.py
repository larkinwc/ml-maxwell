import os, glob, gzip, json
from collections import defaultdict
from vllm import LLM, SamplingParams
p=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
TRACE=os.environ["VLLM_TORCH_PROFILER_DIR"]
TOK=os.environ["TOKV"]
TP=int(os.environ.get("BENCH_TP","1"))
def main():
    # PRODUCTION mode: CUDA graphs ON (no enforce_eager)
    llm=LLM(model=p, tokenizer=TOK, trust_remote_code=True, dtype="float16",
            max_model_len=320, max_num_seqs=1, gpu_memory_utilization=0.90,
            tensor_parallel_size=TP,
            profiler_config={"profiler":"torch","torch_profiler_dir":TRACE})
    llm.generate(["warmup pass one two three"], SamplingParams(temperature=0.0,max_tokens=16))
    llm.start_profile()
    llm.generate(["The capital of France is"], SamplingParams(temperature=0.0,max_tokens=64))
    llm.stop_profile()
    print("PROFILE_DONE")
    # parse the trace for GPU kernel (device) durations
    f=sorted(glob.glob(TRACE+"/rank0.*.json.gz"))[-1]
    d=json.load(gzip.open(f)); ev=d["traceEvents"] if isinstance(d,dict) else d
    # device kernels: ph X with cat kernel OR pid that is a GPU device (args has 'device')
    agg=defaultdict(lambda:[0,0.0])
    for e in ev:
        if not isinstance(e,dict): continue
        cat=e.get("cat","")
        if cat in ("kernel","gpu_op","Kernel"):
            n=e.get("name","?"); agg[n][0]+=1; agg[n][1]+=e.get("dur",0)
    tot=sum(v[1] for v in agg.values())
    print(f"=== GPU KERNELS (graph mode) total={tot/1000:.1f}ms over {len(agg)} kernels ===")
    for n,(c,us) in sorted(agg.items(),key=lambda kv:-kv[1][1])[:30]:
        print(f"  {us/1000:8.2f}ms {100*us/max(tot,1):5.1f}%  n={c:5d}  {n[:75]}")
if __name__=="__main__": main()

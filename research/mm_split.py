import os, glob
from vllm import LLM, SamplingParams
from torch.profiler import profile, ProfilerActivity
p=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
TOK=os.environ["TOKV"]
def count_mm(prof):
    d={}
    for r in prof.key_averages():
        if r.key in ("aten::mm","aten::bmm","aten::addmm"):
            d[r.key]=(r.count, r.self_device_time_total/1000)
    return d
def main():
    llm=LLM(model=p, tokenizer=TOK, trust_remote_code=True, dtype="float16",
            max_model_len=256, max_num_seqs=1, gpu_memory_utilization=0.90,
            enforce_eager=True, tensor_parallel_size=1)
    # long warmup to reach steady decode
    llm.generate(["one two three four five six seven eight"], SamplingParams(temperature=0.0,max_tokens=2))
    # A) measure a generation that is mostly DECODE: 1-token prompt, 32 new tokens
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as pa:
        llm.generate(["Hello"], SamplingParams(temperature=0.0,max_tokens=32))
    print("A) 1-tok prompt + 32 decode steps  mm=",count_mm(pa))
    # B) measure mostly PREFILL: long prompt, 1 new token
    longp=" ".join(["word"]*64)
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as pb:
        llm.generate([longp], SamplingParams(temperature=0.0,max_tokens=1))
    print("B) 64-tok prefill + 1 decode      mm=",count_mm(pb))
if __name__=="__main__": main()

import os, glob
from vllm import LLM, SamplingParams
import torch
p=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
TOK=os.environ["TOKV"]
def main():
    llm=LLM(model=p, tokenizer=TOK, trust_remote_code=True, dtype="float16",
            max_model_len=128, max_num_seqs=1, gpu_memory_utilization=0.90,
            enforce_eager=True, tensor_parallel_size=1)
    llm.generate(["warmup"], SamplingParams(temperature=0.0,max_tokens=4))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                 with_stack=True, record_shapes=True) as prof:
        llm.generate(["The capital of France is"], SamplingParams(temperature=0.0,max_tokens=8))
    # find aten::mm events and print their python stack
    ka=prof.key_averages(group_by_stack_n=8)
    rows=[r for r in ka if r.key=="aten::mm" or "mm" in r.key.lower()]
    for r in sorted(rows,key=lambda x:-x.self_device_time_total)[:6]:
        print("=== ",r.key, "calls=",r.count, "self_cuda_ms=",r.self_device_time_total/1000)
        st=getattr(r,"stack",None) or []
        for s in st[:12]: print("    ",s)
if __name__=="__main__": main()

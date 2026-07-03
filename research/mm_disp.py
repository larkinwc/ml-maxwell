import os, torch, glob
from collections import Counter
from torch.utils._python_dispatch import TorchDispatchMode
from vllm import LLM, SamplingParams
p=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
TOK=os.environ["TOKV"]

cnt=Counter(); active={"on":False}
class Cap(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs=kwargs or {}
        name=str(func)
        if active["on"] and ("mm" in name or "bmm" in name or "addmm" in name or "matmul" in name):
            try:
                a,b=args[0],args[1]
                cnt[(name,tuple(a.shape),tuple(b.shape),str(a.dtype))]+=1
            except: cnt[(name,"?","?","?")]+=1
        return func(*args,**kwargs)

def main():
    llm=LLM(model=p, tokenizer=TOK, trust_remote_code=True, dtype="float16",
            max_model_len=128, max_num_seqs=1, gpu_memory_utilization=0.90,
            enforce_eager=True, tensor_parallel_size=1)
    llm.generate(["warmup"], SamplingParams(temperature=0.0,max_tokens=4))
    with Cap():
        active["on"]=True
        llm.generate(["The capital of France is"], SamplingParams(temperature=0.0,max_tokens=8))
        active["on"]=False
    print("=== aten mm/bmm/addmm during 8-token decode ===")
    for k,v in cnt.most_common(30):
        print(f"  n={v:4d}  {k[0][:30]:30s} A={k[1]} B={k[2]} {k[3]}")
if __name__=="__main__": main()

import os, torch
from collections import Counter
from vllm import LLM, SamplingParams
import glob
p=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
TOK=os.environ["TOKV"]

cnt=Counter()
_mm=torch.Tensor.__matmul__
real_mm=torch.mm; real_matmul=torch.matmul; real_bmm=torch.bmm
def logshape(tag,a,b):
    try: cnt[(tag,tuple(a.shape),tuple(b.shape),str(a.dtype))]+=1
    except: pass
def mm(a,b,*A,**K): logshape("mm",a,b); return real_mm(a,b,*A,**K)
def matmul(a,b,*A,**K): logshape("matmul",a,b); return real_matmul(a,b,*A,**K)
def bmm(a,b,*A,**K): logshape("bmm",a,b); return real_bmm(a,b,*A,**K)
torch.mm=mm; torch.matmul=matmul; torch.bmm=bmm

def main():
    llm=LLM(model=p, tokenizer=TOK, trust_remote_code=True, dtype="float16",
            max_model_len=128, max_num_seqs=1, gpu_memory_utilization=0.90,
            enforce_eager=True, tensor_parallel_size=1)
    llm.generate(["warmup"], SamplingParams(temperature=0.0,max_tokens=4))
    cnt.clear()
    llm.generate(["The capital of France is"], SamplingParams(temperature=0.0,max_tokens=8))
    print("=== matmul shapes during 8-token decode (per call counts) ===")
    for k,v in cnt.most_common(25):
        print(f"  n={v:4d}  {k[0]:7s} A={k[1]} B={k[2]} {k[3]}")
if __name__=="__main__": main()

import os, torch, glob
from collections import Counter
from vllm import LLM, SamplingParams
p=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
TOK=os.environ["TOKV"]
def main():
    llm=LLM(model=p, tokenizer=TOK, trust_remote_code=True, dtype="float16",
            max_model_len=128, max_num_seqs=1, gpu_memory_utilization=0.90,
            enforce_eager=True, tensor_parallel_size=1)
    m=llm.llm_engine.model_executor.driver_worker.model_runner.model
    fp16=Counter(); fp16_bytes=0; total_bytes=0
    big=[]
    for name,param in m.named_parameters():
        b=param.numel()*param.element_size(); total_bytes+=b
        if param.dtype in (torch.float16,torch.bfloat16,torch.float32):
            # only count 2D weight-ish (matmul) params, skip norms (<1M)
            if param.dim()>=2 and param.numel()>=1_000_000:
                fp16[(tuple(param.shape),str(param.dtype))]+=1; fp16_bytes+=b
                big.append((b,name,tuple(param.shape),str(param.dtype)))
    print(f"TOTAL param bytes={total_bytes/1e9:.2f}GB  fp16/fp32 2D-big bytes={fp16_bytes/1e9:.3f}GB")
    print("=== large fp16/fp32 2D params (matmul candidates, per-layer dedup) ===")
    seen=Counter()
    for b,name,shp,dt in big:
        role=".".join(x for x in name.split(".") if not x.isdigit())
        seen[(role,shp,dt)]+=1
    for (role,shp,dt),c in sorted(seen.items(),key=lambda kv:-kv[0][1][0]*kv[0][1][1]):
        print(f"  n={c:3d}  {role:55s} {shp} {dt}  ({shp[0]*shp[1]*2/1e6:.1f}MB ea)")
if __name__=="__main__": main()

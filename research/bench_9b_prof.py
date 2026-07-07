import time, os
from vllm import LLM, SamplingParams
p="/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_0.gguf"
TP=int(os.environ["BENCH_TP"])
EAGER=os.environ.get("BENCH_EAGER","1")=="1"
def main():
    llm=LLM(model=p, tokenizer="Qwen/Qwen3.5-9B", trust_remote_code=True, dtype="float16",
            max_model_len=320, max_num_seqs=1, gpu_memory_utilization=0.90,
            enforce_eager=EAGER, tensor_parallel_size=TP)
    # warmup to cover triton JIT + (if enabled) cudagraph capture
    llm.generate(["Hello there friend, how are you"], SamplingParams(temperature=0.0,max_tokens=16))
    llm.generate(["Second warmup pass for graph replay"], SamplingParams(temperature=0.0,max_tokens=16))
    sp=SamplingParams(temperature=0.0, max_tokens=128)
    t0=time.time()
    out=llm.generate(["The capital of France is"], sp)
    dt=time.time()-t0
    n=len(out[0].outputs[0].token_ids)
    print(f"RESULT TP={TP} EAGER={EAGER}: {n} tokens in {dt:.2f}s = {n/dt:.2f} tok/s")
    print("TEXT:", repr(out[0].outputs[0].text[:100]))
if __name__=="__main__": main()

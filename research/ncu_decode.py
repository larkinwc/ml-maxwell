import os
from vllm import LLM, SamplingParams
p="/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_0.gguf"
def main():
    llm=LLM(model=p, tokenizer="__TOK__", trust_remote_code=True, dtype="float16",
            max_model_len=320, max_num_seqs=1, gpu_memory_utilization=0.90,
            enforce_eager=True, tensor_parallel_size=1)
    # tiny generation; ncu will sample the mul_mat_vec kernel launches
    llm.generate(["The capital of France is"], SamplingParams(temperature=0.0,max_tokens=4))
    print("NCU_DECODE_DONE")
if __name__=="__main__": main()

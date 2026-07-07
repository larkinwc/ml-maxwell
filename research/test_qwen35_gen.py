import os, time
from vllm import LLM, SamplingParams

p = "/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_0.gguf"


def main():
    llm = LLM(
        model=p,
        tokenizer="Qwen/Qwen3.5-0.8B",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    prompts = [
        "The capital of France is",
        "Q: What is 2+2? A:",
        "Write one sentence about the ocean:",
    ]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    for o in outs:
        print("PROMPT:", repr(o.prompt))
        print("OUTPUT:", repr(o.outputs[0].text))
        print("---")
    print(f"elapsed {dt:.1f}s")


if __name__ == "__main__":
    main()

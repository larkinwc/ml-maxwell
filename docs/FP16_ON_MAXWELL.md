# The fp16-arithmetic wall on Maxwell (sm_50) — and how to clear it

## The problem (discovered building vLLM v0.23 csrc)

Maxwell GM107 (Tesla M10, sm_50) has fp16 **storage + conversion** but **no fp16
ALU**. `ptxas` rejects fp16 arithmetic PTX for sm_50:

```
ptxas paged_attention_v1.compute_50.ptx error:
  Feature 'f16 arithmetic and compare instructions' requires .target sm_53 or higher
```

This is **systemic**, not one kernel. vLLM's `csrc/attention/dtype_float16.cuh`
implements fp16 math with inline PTX (`add.f16`, `mul.f16`, `fma.rn.f16x2`), and the
same `__hfma2`/`__hadd2`/half-`atomicAdd` family appears in MoE (`moe_wna16.cu`),
GPTQ (`q_gemm.cu`, `qdq_*.cuh`), activation, and sampler kernels.

Hardware reality (confirmed by NVIDIA docs + StackOverflow + ptxas):
- fp16 **arithmetic** intrinsics (`__hadd/__hmul/__hfma`, packed `*2`) require **sm_53+**
- `__dp4a` (byte dot product, used by INT8/quant) requires **sm_61+**
- sm_50 **can** do `__half2float` / `__float2half` conversion, and native fp32 math

## The proven solution: ggml/llama.cpp's tiered fallback

llama.cpp runs on these exact M10s. Its `ggml/src/ggml-cuda/common.cuh` is the
reference. The pattern:

```cpp
#define GGML_CUDA_CC_PASCAL 600
#define GGML_CUDA_CC_DP4A   610   // min cc for __dp4a

// fp16 ALU gated at Pascal; Maxwell falls to the emulation branch
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
  #define FP16_AVAILABLE
#endif
#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
  #define FAST_FP16_AVAILABLE
#endif
```

Multiply-add, the core op, has three tiers — **the emulation branch is what we
reuse for Maxwell**:

```cpp
static __device__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v*u);   // native half mul
    acc += tmp.x + tmp.y;
#else                                          // <-- MAXWELL path
    const float2 tmpv = __half22float2(v);     // convert up
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x*tmpu.x;                       // fp32 math
    acc += tmpv.y*tmpu.y;
#endif
}
```

INT8 dot product (`__dp4a`) has the same idea — emulate with scalar int8 mults when
`__CUDA_ARCH__ < 610`. ggml ships this; it's how quant runs on Maxwell.

## Strategy for vLLM kernels

Mirror ggml: introduce a small `maxwell_compat.cuh` in the fork defining
`VLLM_FP16_HW` / `VLLM_DP4A_HW` arch gates, then for each needed kernel family
replace native-fp16 ops with convert→fp32→compute when the gate is off:

1. **Attention (priority 1)** — `dtype_float16.cuh`: the FloatAdd/Mul/Fma helpers
   already centralize the PTX. Provide sm_50 variants that convert to fp32, compute,
   convert back. This is *localized* — kernels call the helpers, don't inline PTX.
2. **Activation / sampler** — same convert-up treatment where half math is used.
3. **Quant (GPTQ/AWQ/WNA16)** — needs both the fp16 emulation AND `__dp4a`
   emulation (scalar int8). Mine ggml's `ggml_cuda_dp4a` + the exllama/AWQ origins.

## Perf expectation
Each emulated fp16 op = convert + fp32 op + convert (≈2-3× the instructions). On a
decode tier that's **latency/host-RAM bound anyway** (see NCCL findings), the ALU
overhead is unlikely to dominate. fp32 accumulation is also *more* accurate.

## Source references
- llama.cpp `ggml/src/ggml-cuda/common.cuh` (FP16_AVAILABLE / FAST_FP16_AVAILABLE,
  `ggml_cuda_mad`, `ggml_cuda_dp4a`) — the canonical tiered fallback.
- We already built `libggml-cuda.so` for sm_50 (`artifacts/llama.cpp-sm50/`),
  proving the pattern compiles+runs on our M10s.
- exllama / exllamav2 — handwritten GPTQ kernels, origin of vLLM's `qdq_*.cuh`;
  check their pre-sm_61 paths for the quant dequant fallback.

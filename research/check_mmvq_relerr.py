import torch, numpy as np, gguf, glob
from gguf.quants import dequantize
from vllm import _custom_ops as ops

# Pull a real Q4_0 weight tensor from the 9B gguf and compare MMVQ vs dequant-matmul.
gp=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
r=gguf.GGUFReader(gp)
# find a 2D Q4_0 tensor (a linear weight)
t=None
for x in r.tensors:
    if x.tensor_type==gguf.GGMLQuantizationType.Q4_0 and len(x.shape)==2:
        t=x; break
assert t is not None, "no Q4_0 2D tensor"
shape=tuple(reversed([int(s) for s in t.shape]))  # [nrows, ncols]
nrows, ncols = shape
print(f"tensor {t.name} shape={shape} type=Q4_0")

qweight=torch.tensor(np.array(t.data)).cuda()                 # raw quant bytes
qtype=torch.tensor(gguf.GGMLQuantizationType.Q4_0.value).cuda()
ref_w=torch.tensor(dequantize(np.array(t.data), t.tensor_type).astype(np.float32)).reshape(shape).cuda()  # [nrows,ncols] fp32

torch.manual_seed(0)
x=torch.randn(1, ncols, dtype=torch.float16, device="cuda")   # single decode vector
# MMVQ path
y_mmvq=ops.ggml_mul_mat_vec_a8(qweight, x, qtype, nrows).float().flatten()
# reference: x @ W^T
y_ref=(x.float() @ ref_w.t()).flatten()

abs_err=(y_mmvq-y_ref).abs()
rel=abs_err/(y_ref.abs()+1e-3)
print(f"MMVQ vs dequant-matmul: max_abs={abs_err.max():.4f} mean_abs={abs_err.mean():.4f} "
      f"max_rel={rel.max():.4f} mean_rel={rel.mean():.4f}")
# Q4_0 MMVQ quant noise is ~1e-2; flag if structurally broken
print("VERDICT:", "OK" if rel.mean()<0.05 else "SUSPECT")

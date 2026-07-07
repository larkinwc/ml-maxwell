import torch, numpy as np, gguf, glob
from vllm import _custom_ops as ops
gp=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
r=gguf.GGUFReader(gp)
Q4_0=gguf.GGMLQuantizationType.Q4_0
qtype=torch.tensor(Q4_0.value).cuda()

# pick the largest Q4_0 2D tensor
best=None
for x in r.tensors:
    if x.tensor_type==Q4_0 and len(x.shape)==2:
        shape=tuple(reversed([int(s) for s in x.shape]))
        if best is None or shape[0]*shape[1]>best[1][0]*best[1][1]:
            best=(x.name,shape,x)
name,shape,xt=best
nrows,ncols=shape
data=torch.tensor(np.array(xt.data)).cuda()
wbytes=data.numel()
print(f"largest tensor {name} shape={shape} bytes={wbytes/1e6:.1f}MB")
xv=torch.randn(1,ncols,dtype=torch.float16,device='cuda')

# warm
for _ in range(5): ops.ggml_mul_mat_vec_a8(data,xv,qtype,nrows)
torch.cuda.synchronize()
e0=torch.cuda.Event(True); e1=torch.cuda.Event(True)
IT=200
e0.record()
for _ in range(IT): ops.ggml_mul_mat_vec_a8(data,xv,qtype,nrows)
e1.record(); torch.cuda.synchronize()
ms=e0.elapsed_time(e1)/IT
print(f"single op: {ms:.4f} ms, {wbytes/(ms*1e-3)/1e9:.1f} GB/s  ({wbytes/(ms*1e-3)/1e9/73*100:.0f}% of 73)")

# also CUDA-graph capture to remove launch overhead
g=torch.cuda.CUDAGraph()
torch.cuda.synchronize()
with torch.cuda.graph(g):
    out=ops.ggml_mul_mat_vec_a8(data,xv,qtype,nrows)
for _ in range(5): g.replay()
torch.cuda.synchronize()
e0.record()
for _ in range(IT): g.replay()
e1.record(); torch.cuda.synchronize()
msg=e0.elapsed_time(e1)/IT
print(f"graphed op: {msg:.4f} ms, {wbytes/(msg*1e-3)/1e9:.1f} GB/s  ({wbytes/(msg*1e-3)/1e9/73*100:.0f}% of 73)")
print(f"launch overhead per call: {(ms-msg)*1e3:.1f} us")

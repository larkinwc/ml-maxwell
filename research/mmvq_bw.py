import torch, time, numpy as np, gguf, glob
from vllm import _custom_ops as ops

gp=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
r=gguf.GGUFReader(gp)

Q4_0=gguf.GGMLQuantizationType.Q4_0
qtype=torch.tensor(Q4_0.value).cuda()

# Collect all Q4_0 2D weight tensors (the linear layers that go through MMVQ).
tensors=[]
total_bytes=0
for x in r.tensors:
    if x.tensor_type==Q4_0 and len(x.shape)==2:
        shape=tuple(reversed([int(s) for s in x.shape]))  # [nrows, ncols]
        data=torch.tensor(np.array(x.data)).cuda()
        tensors.append((x.name, shape, data))
        total_bytes += data.numel()  # bytes (uint8)

print(f"#Q4_0 2D tensors={len(tensors)} total_weight_bytes={total_bytes/1e9:.3f} GB")

# Build a single-token activation per tensor (x: [1, ncols] fp16)
acts=[(torch.randn(1, shape[1], dtype=torch.float16, device='cuda'), shape[0]) for _,shape,_ in tensors]

def run_all():
    outs=[]
    for (name,shape,data),(xv,nrows) in zip(tensors,acts):
        outs.append(ops.ggml_mul_mat_vec_a8(data, xv, qtype, nrows))
    return outs

# warmup
for _ in range(3): run_all()
torch.cuda.synchronize()

ITERS=20
t0=time.time()
for _ in range(ITERS): run_all()
torch.cuda.synchronize()
dt=(time.time()-t0)/ITERS

# These tensors appear once per token but the full model reads ALL layers' weights.
# Here we only loaded the tensors present in the gguf (full model). So one run_all == one token's MMVQ weight reads.
gbps = total_bytes/dt/1e9
print(f"MMVQ-only: {dt*1e3:.2f} ms/token-equiv, weight_bytes={total_bytes/1e9:.3f} GB -> {gbps:.1f} GB/s achieved")
print(f"HBM ceiling ~73 GB/s -> MMVQ efficiency = {gbps/73*100:.0f}%")
print(f"If MMVQ were the ONLY cost: {1.0/dt:.2f} tok/s")

import torch, time, numpy as np, gguf, glob
from vllm import _custom_ops as ops
gp=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
r=gguf.GGUFReader(gp)
# Q6_K = output.weight (lm_head); Q5_K = ssm_out
for want in ["Q6_K","Q5_K"]:
    t=None
    for x in r.tensors:
        if x.tensor_type.name==want and len(x.shape)==2:
            t=x; break
    qt=int(t.tensor_type.value)
    nrows,ncols=tuple(reversed([int(s) for s in t.shape]))
    data=torch.tensor(np.array(t.data)).cuda()
    wbytes=data.numel()
    qtype=torch.tensor(qt).cuda()
    x=torch.randn(1,ncols,dtype=torch.float16,device='cuda')
    for _ in range(5): ops.ggml_mul_mat_vec_a8(data,x,qtype,nrows)
    torch.cuda.synchronize()
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True)
    IT=50
    e0.record()
    for _ in range(IT): ops.ggml_mul_mat_vec_a8(data,x,qtype,nrows)
    e1.record();torch.cuda.synchronize()
    ms=e0.elapsed_time(e1)/IT
    print(f"{want}: {t.name} shape=({nrows},{ncols}) {wbytes/1e6:.0f}MB  {ms:.3f}ms  {wbytes/(ms*1e-3)/1e9:.1f} GB/s  ({wbytes/(ms*1e-3)/1e9/73*100:.0f}% of 73)")

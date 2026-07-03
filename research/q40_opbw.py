import torch, numpy as np, gguf, glob
from vllm import _custom_ops as ops
gp=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
r=gguf.GGUFReader(gp)
def bench(want, namefilter=None):
    t=None
    for x in r.tensors:
        if x.tensor_type.name==want and len(x.shape)==2 and (namefilter is None or namefilter in x.name):
            t=x; break
    qt=int(t.tensor_type.value)
    nrows,ncols=tuple(reversed([int(s) for s in t.shape]))
    data=torch.tensor(np.array(t.data)).cuda()
    wbytes=data.numel()
    qtype=torch.tensor(qt).cuda()
    x=torch.randn(1,ncols,dtype=torch.float16,device='cuda')
    for _ in range(5): ops.ggml_mul_mat_vec_a8(data,x,qtype,nrows)
    torch.cuda.synchronize()
    e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);IT=80
    e0.record()
    for _ in range(IT): ops.ggml_mul_mat_vec_a8(data,x,qtype,nrows)
    e1.record();torch.cuda.synchronize()
    ms=e0.elapsed_time(e1)/IT
    print(f"{want:6s} {t.name:28s} ({nrows},{ncols}) {wbytes/1e6:6.0f}MB  {ms:.3f}ms  {wbytes/(ms*1e-3)/1e9:5.1f} GB/s  ({wbytes/(ms*1e-3)/1e9/73*100:3.0f}% of 73)")
# compare same (4096,4096) shape across quant types where possible
bench("Q4_0","ffn_gate")     # 4096-ish Q4_0
bench("Q5_K","ssm_out")      # 4096x4096 Q5_K
bench("Q6_K","output")       # lm_head Q6_K
bench("Q4_1","ffn_down")     # Q4_1

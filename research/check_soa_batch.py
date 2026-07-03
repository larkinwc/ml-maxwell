import torch, numpy as np, gguf, glob
from vllm import _custom_ops as ops
gp=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
r=gguf.GGUFReader(gp)
Q4_0=gguf.GGMLQuantizationType.Q4_0
qtype=torch.tensor(Q4_0.value).cuda()
for x in r.tensors:
    if x.tensor_type==Q4_0 and len(x.shape)==2 and "attn_output" in x.name:
        t=x; break
nrows,ncols=tuple(reversed([int(s) for s in t.shape])); bpr=ncols//32
qweight=torch.tensor(np.array(t.data)).view(nrows,bpr*18).cuda()
wb=qweight.view(nrows,bpr,18)
quants=wb[:,:,2:18].contiguous().view(nrows,bpr*16)
scales=wb[:,:,0:2].contiguous().view(nrows,bpr*2).view(torch.float16).contiguous()
print(f"{t.name} nrows={nrows} ncols={ncols}")
torch.manual_seed(0)
for vecs in [1,4,8,16,32]:
    x=torch.randn(vecs,ncols,dtype=torch.float16,device="cuda")
    y_aos=ops.ggml_mul_mat_vec_a8(qweight,x,qtype,nrows).float()
    y_soa=ops.ggml_mul_mat_vec_a8_soa(quants,scales,x,nrows).float()
    d=(y_aos-y_soa).abs()
    print(f"vecs={vecs}: max_abs={d.max():.5f} shape={tuple(y_soa.shape)} ok={(d.max()<1e-2).item()}")

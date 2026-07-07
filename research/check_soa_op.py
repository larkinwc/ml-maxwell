import torch, numpy as np, gguf, glob
from vllm import _custom_ops as ops

gp=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
r=gguf.GGUFReader(gp)
Q4_0=gguf.GGMLQuantizationType.Q4_0
qtype=torch.tensor(Q4_0.value).cuda()

# pick a representative non-embedding Q4_0 2D weight
t=None
for x in r.tensors:
    if x.tensor_type==Q4_0 and len(x.shape)==2 and "attn_k" in x.name:
        t=x; break
nrows,ncols=tuple(reversed([int(s) for s in t.shape]))
bpr=ncols//32
raw=np.array(t.data)
qweight=torch.tensor(raw).view(nrows, bpr*18).cuda()
print(f"tensor {t.name} nrows={nrows} ncols={ncols} bpr={bpr}")

# build SoA buffers (mirror gguf.py _maybe_build_q4_0_soa)
wb=qweight.view(nrows,bpr,18)
quants=wb[:,:,2:18].contiguous().view(nrows,bpr*16)
scales=wb[:,:,0:2].contiguous().view(nrows,bpr*2).view(torch.float16).contiguous()

torch.manual_seed(0)
for vecs in [1,2]:
    x=torch.randn(vecs, ncols, dtype=torch.float16, device='cuda')
    y_aos=ops.ggml_mul_mat_vec_a8(qweight, x, qtype, nrows).float()
    y_soa=ops.ggml_mul_mat_vec_a8_soa(quants, scales, x, nrows).float()
    diff=(y_aos-y_soa).abs()
    rel=diff/(y_aos.abs()+1e-3)
    print(f"vecs={vecs}: max_abs={diff.max():.5f} mean_abs={diff.mean():.6f} "
          f"max_rel={rel.max():.4f} mean_rel={rel.mean():.5f}")
    print("  VERDICT:", "OK (matches AoS)" if diff.max()<1e-2 else "MISMATCH")

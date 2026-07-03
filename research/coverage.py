import numpy as np, gguf, glob
gp=glob.glob("/home/larkinwc/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_0.gguf")[0]
r=gguf.GGUFReader(gp)
Q4_0=gguf.GGMLQuantizationType.Q4_0
# Sharded/fused in vLLM: qkv (attn q/k/v), gate_up (ffn gate+up), in_proj qkvz (GDN)
# Non-sharded (SoA-eligible): o_proj/out_proj, down_proj, lm_head/output, others
sharded_keys=["attn_q","attn_k","attn_v","ffn_gate","ffn_up","in_proj","_qkv","attn_qkv"]
soa=0; aos=0; total=0
soa_names=set(); aos_names=set()
for x in r.tensors:
    if x.tensor_type!=Q4_0 or len(x.shape)!=2: continue
    b=int(np.array(x.data).size)
    total+=b
    if any(k in x.name for k in sharded_keys):
        aos+=b; aos_names.add(x.name.split(".")[-2] if "." in x.name else x.name)
    else:
        soa+=b; soa_names.add(x.name.split(".")[-2] if "." in x.name else x.name)
print(f"total Q4_0 bytes={total/1e9:.3f} GB")
print(f"SoA-eligible (non-sharded)={soa/1e9:.3f} GB ({soa/total*100:.0f}%)  layers={sorted(soa_names)}")
print(f"AoS (sharded/fused)       ={aos/1e9:.3f} GB ({aos/total*100:.0f}%)  layers={sorted(aos_names)}")

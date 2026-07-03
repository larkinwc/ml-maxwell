import torch, time
# Measure achievable HBM read+write bandwidth on one M10 die via large contiguous ops.
dev="cuda:0"
N = 512*1024*1024  # 512M float16 = 1 GiB
a = torch.empty(N, dtype=torch.float16, device=dev).normal_()
b = torch.empty_like(a)

torch.cuda.synchronize()
# copy: reads N*2 bytes + writes N*2 bytes
for _ in range(3):
    b.copy_(a)
torch.cuda.synchronize()
t0=time.time()
ITERS=30
for _ in range(ITERS):
    b.copy_(a)
torch.cuda.synchronize()
dt=time.time()-t0
bytes_moved = ITERS * N * 2 * 2  # read + write, fp16
print(f"copy: {dt/ITERS*1e3:.2f} ms/iter, {bytes_moved/dt/1e9:.1f} GB/s (read+write)")

# pure read via reduction (sum): reads N*2 bytes
torch.cuda.synchronize()
t0=time.time()
for _ in range(ITERS):
    s=a.sum()
torch.cuda.synchronize()
dt=time.time()-t0
bytes_read = ITERS * N * 2
print(f"sum(read): {dt/ITERS*1e3:.2f} ms/iter, {bytes_read/dt/1e9:.1f} GB/s (read-only)")
print("device:", torch.cuda.get_device_name(0))

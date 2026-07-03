// Standalone bandwidth microbenchmark: Q4_0 weight read patterns on Maxwell.
// Compares current AoS 18-byte-block access vs SoA repack with 128-bit (int4) loads.
// Build: nvcc -arch=sm_50 -O3 q40_bw.cu -o q40_bw
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

#define QK4_0 32
#define WARP 32

struct __align__(2) block_q4_0 { half d; uint8_t qs[QK4_0/2]; }; // 18 bytes

__device__ __forceinline__ int dp4a_sw(int a,int b,int c){
  int s=c;
  s+=((a<<24)>>24)*((b<<24)>>24); s+=((a<<16)>>24)*((b<<16)>>24);
  s+=((a<<8)>>24)*((b<<8)>>24);   s+=(a>>24)*(b>>24);
  return s;
}

// ---- Kernel A: current AoS pattern (mirrors mul_mat_vec_q for Q4_0, vdr=2) ----
__global__ void mmvq_aos(const block_q4_0* __restrict__ x, const int* __restrict__ y,
                         float* __restrict__ dst, int ncols, int nrows){
  int row = blockIdx.x*blockDim.y + threadIdx.y;
  if(row>=nrows) return;
  int bpr = ncols/QK4_0;          // blocks per row
  int bpw = 2*WARP/(QK4_0/8);     // qi=QK4_0/8=4 -> bpw=2*32/4=16
  int qi = QK4_0/8;               // 4
  float tmp=0.f;
  for(int i=threadIdx.x/(qi/2); i<bpr; i+=bpw){
    const block_q4_0* b=&x[row*bpr+i];
    int iqs=2*(threadIdx.x%(qi/2));
    // two 4-byte loads from qs (mirrors get_int_from_uint8 x2)
    const uint16_t* q16=(const uint16_t*)(b->qs+sizeof(int)*iqs);
    int v0=q16[0]|(q16[1]<<16);
    const uint16_t* q16b=(const uint16_t*)(b->qs+sizeof(int)*(iqs+1));
    int v1=q16b[0]|(q16b[1]<<16);
    float d=__half2float(b->d);
    tmp += d*(dp4a_sw(v0,y[(i*4+iqs)&255],0)+dp4a_sw(v1,y[(i*4+iqs+1)&255],0));
  }
  #pragma unroll
  for(int m=WARP/2;m>0;m>>=1) tmp+=__shfl_xor_sync(0xffffffff,tmp,m);
  if(threadIdx.x==0) dst[row]=tmp;
}

// ---- Kernel B: SoA repack, 128-bit int4 loads ----
// quants: contiguous uint8[nrows*bpr*16], block b at qs+ b*16 (16-byte aligned)
// scales: contiguous half[nrows*bpr]
__global__ void mmvq_soa(const uint4* __restrict__ quants, const half* __restrict__ scales,
                         const int* __restrict__ y, float* __restrict__ dst, int ncols, int nrows){
  int row = blockIdx.x*blockDim.y + threadIdx.y;
  if(row>=nrows) return;
  int bpr = ncols/QK4_0;
  float tmp=0.f;
  // one thread per block, warp strides 32 blocks at a time (32*16=512B contiguous load)
  for(int i=threadIdx.x; i<bpr; i+=WARP){
    uint4 q = quants[row*bpr+i];   // 128-bit aligned load of all 16 quant bytes
    float d=__half2float(scales[row*bpr+i]);
    const int* qi=(const int*)&q;  // 4 ints = 16 bytes = 32 nibbles
    int acc=0;
    #pragma unroll
    for(int k=0;k<4;k++) acc=dp4a_sw(qi[k], y[(i*4+k)&255], acc);
    tmp += d*acc;
  }
  #pragma unroll
  for(int m=WARP/2;m>0;m>>=1) tmp+=__shfl_xor_sync(0xffffffff,tmp,m);
  if(threadIdx.x==0) dst[row]=tmp;
}

int main(){
  int nrows=16384, ncols=4096;
  int bpr=ncols/QK4_0;            // 128
  size_t nblocks=(size_t)nrows*bpr;
  size_t aos_bytes=nblocks*sizeof(block_q4_0); // 18B each
  size_t quant_bytes=nblocks*16, scale_bytes=nblocks*2;

  block_q4_0* d_aos; cudaMalloc(&d_aos,aos_bytes);
  uint4* d_q; cudaMalloc(&d_q,quant_bytes);
  half* d_s; cudaMalloc(&d_s,scale_bytes);
  int* d_y; cudaMalloc(&d_y,256*sizeof(int));
  float* d_dst; cudaMalloc(&d_dst,nrows*sizeof(float));
  cudaMemset(d_aos,1,aos_bytes); cudaMemset(d_q,1,quant_bytes);
  cudaMemset(d_s,1,scale_bytes); cudaMemset(d_y,1,256*sizeof(int));

  dim3 blk(WARP,4,1);  // MMV_Y-like, 4 rows/block
  dim3 grdA((nrows+3)/4,1,1);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  int IT=200;

  // warmup + time AoS
  for(int i=0;i<10;i++) mmvq_aos<<<grdA,blk>>>(d_aos,d_y,d_dst,ncols,nrows);
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  for(int i=0;i<IT;i++) mmvq_aos<<<grdA,blk>>>(d_aos,d_y,d_dst,ncols,nrows);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float msA; cudaEventElapsedTime(&msA,a,b); msA/=IT;
  double gbA=aos_bytes/ (msA*1e-3)/1e9;
  printf("AoS  : %.3f ms, %.1f GB/s (read %.1f MB)\n", msA, gbA, aos_bytes/1e6);

  // time SoA
  for(int i=0;i<10;i++) mmvq_soa<<<grdA,blk>>>(d_q,d_s,d_y,d_dst,ncols,nrows);
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  for(int i=0;i<IT;i++) mmvq_soa<<<grdA,blk>>>(d_q,d_s,d_y,d_dst,ncols,nrows);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float msB; cudaEventElapsedTime(&msB,a,b); msB/=IT;
  double gbB=(quant_bytes+scale_bytes)/(msB*1e-3)/1e9;
  printf("SoA  : %.3f ms, %.1f GB/s (read %.1f MB)\n", msB, gbB,(quant_bytes+scale_bytes)/1e6);
  printf("speedup (time): %.2fx\n", msA/msB);
  return 0;
}

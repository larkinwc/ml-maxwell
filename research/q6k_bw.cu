// Q6_K MMVQ *bandwidth* harness (pure read pattern; math irrelevant for BW).
// AoS: real 210-byte super-block, strided per-thread reads like vec_dot_q6_K_q8_1.
// SoA: ql/qh/scales separated into contiguous 128-bit-aligned arrays (uint4 loads).
// Build: nvcc -arch=sm_50 -O3 q6k_bw.cu -o q6k_bw
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#define QK_K 256
#define WARP 32

struct __align__(2) block_q6_K {
  uint8_t ql[QK_K/2];      // 128
  uint8_t qh[QK_K/4];      // 64
  int8_t  scales[QK_K/16]; // 16
  half    d;               // 2  => 210 bytes
};

__device__ __forceinline__ int gi_u8(const uint8_t*x8,int i){
  const uint16_t*x16=(const uint16_t*)(x8+4*i); int x=0; x|=x16[0]; x|=x16[1]<<16; return x;
}

// ---- AoS: read every byte of the super-block with 4-byte scalar loads ----
// 32 threads cover the block: ql is 128B=32 ints, qh 64B=16 ints, scales 16B=4 ints.
template<int MY>
__global__ void aos(const block_q6_K* __restrict__ x, float* __restrict__ dst,
                    int ncols, int nrows){
  int row=blockIdx.x*MY+threadIdx.y; if(row>=nrows)return;
  int bpr=ncols/QK_K; int t=threadIdx.x; int acc=0;
  for(int b=0;b<bpr;b++){
    const block_q6_K* bq=&x[row*bpr+b];
    acc += gi_u8(bq->ql, t);            // 32 ints -> all 128B of ql (uint16 pairs, 2-byte safe)
    if(t<16) acc += gi_u8(bq->qh, t);   // 16 ints -> all 64B of qh
    if(t<16) acc += (int)bq->scales[t]; // 16 int8 scales (byte reads, safe)
    if(t==0) acc += __half2int_rn(bq->d);
  }
  #pragma unroll
  for(int m=WARP/2;m>0;m>>=1) acc+=__shfl_xor_sync(0xffffffff,acc,m);
  if(t==0) dst[row]=acc;
}

// ---- SoA: ql[nrows*bpr*128], qh[*64], scales[*16], d[*] contiguous ----
// Loaded as uint4 (128-bit). Warp strides whole blocks; each thread loads parts.
template<int MY>
__global__ void soa(const uint4* __restrict__ ql,  // 8 uint4 per block (128B)
                    const uint4* __restrict__ qh,  // 4 uint4 per block (64B)
                    const uint4* __restrict__ sc,  // 1 uint4 per block (16B)
                    const half*  __restrict__ d,
                    float* __restrict__ dst, int ncols, int nrows){
  int row=blockIdx.x*MY+threadIdx.y; if(row>=nrows)return;
  int bpr=ncols/QK_K; int t=threadIdx.x; int acc=0;
  for(int b=0;b<bpr;b++){
    size_t bi=(size_t)row*bpr+b;
    // 8 uint4 of ql across threads 0..7, 4 of qh across 8..11, 1 of scales at 12
    if(t<8){ uint4 v=ql[bi*8 + t]; acc+=v.x+v.y+v.z+v.w; }
    else if(t<12){ uint4 v=qh[bi*4 + (t-8)]; acc+=v.x+v.y+v.z+v.w; }
    else if(t==12){ uint4 v=sc[bi]; acc+=v.x+v.y+v.z+v.w; }
    else if(t==13){ acc+=__half2int_rn(d[bi]); }
  }
  #pragma unroll
  for(int m=WARP/2;m>0;m>>=1) acc+=__shfl_xor_sync(0xffffffff,acc,m);
  if(t==0) dst[row]=acc;
}

int main(){
  int nrows=8192, ncols=4096;        // lm_head-ish shape chunk
  int bpr=ncols/QK_K;                // 16
  size_t nb=(size_t)nrows*bpr;
  size_t aos_bytes=nb*210;
  size_t qlb=nb*128, qhb=nb*64, scb=nb*16, db=nb*2;
  block_q6_K* d_aos; cudaMalloc(&d_aos,aos_bytes); cudaMemset(d_aos,1,aos_bytes);
  uint4 *d_ql,*d_qh,*d_sc; half* d_d; float* d_dst;
  cudaMalloc(&d_ql,qlb); cudaMalloc(&d_qh,qhb); cudaMalloc(&d_sc,scb);
  cudaMalloc(&d_d,db); cudaMalloc(&d_dst,nrows*4);
  cudaMemset(d_ql,1,qlb); cudaMemset(d_qh,1,qhb); cudaMemset(d_sc,1,scb); cudaMemset(d_d,1,db);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); int IT=300;
  #define BENCH(NAME,MY,CALL,BYTES) do{ \
    dim3 blk(WARP,MY,1),grd((nrows+MY-1)/MY,1,1); \
    for(int i=0;i<10;i++){CALL;} cudaError_t e=cudaDeviceSynchronize(); \
    if(e){printf("%s MY=%d LAUNCH ERR: %s\n",NAME,MY,cudaGetErrorString(e));break;} \
    cudaEventRecord(a); for(int i=0;i<IT;i++){CALL;} cudaEventRecord(b);cudaEventSynchronize(b); \
    float ms;cudaEventElapsedTime(&ms,a,b);ms/=IT; \
    printf("%-12s MY=%d: %.4f ms, %.1f GB/s (%.0f%% of 73)\n",NAME,MY,ms,(BYTES)/(ms*1e-3)/1e9,(BYTES)/(ms*1e-3)/1e9/73*100);}while(0)
  BENCH("AoS",1,(aos<1><<<grd,blk>>>(d_aos,d_dst,ncols,nrows)),(double)aos_bytes);
  BENCH("AoS",2,(aos<2><<<grd,blk>>>(d_aos,d_dst,ncols,nrows)),(double)aos_bytes);
  BENCH("AoS",4,(aos<4><<<grd,blk>>>(d_aos,d_dst,ncols,nrows)),(double)aos_bytes);
  BENCH("SoA",1,(soa<1><<<grd,blk>>>(d_ql,d_qh,d_sc,d_d,d_dst,ncols,nrows)),(double)(qlb+qhb+scb+db));
  BENCH("SoA",2,(soa<2><<<grd,blk>>>(d_ql,d_qh,d_sc,d_d,d_dst,ncols,nrows)),(double)(qlb+qhb+scb+db));
  BENCH("SoA",4,(soa<4><<<grd,blk>>>(d_ql,d_qh,d_sc,d_d,d_dst,ncols,nrows)),(double)(qlb+qhb+scb+db));
  return 0;
}

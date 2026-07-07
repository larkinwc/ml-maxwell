// Faithful Q4_0 MMVQ bandwidth harness. Mirrors real vec_dot_q4_0_q8_1 incl. q8_1 y reads.
// Build: nvcc -arch=sm_50 -O3 q40_bw2.cu -o q40_bw2
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#define QK4_0 32
#define WARP 32

struct __align__(2) block_q4_0 { half d; uint8_t qs[QK4_0/2]; };       // 18B
struct __align__(4) block_q8_1 { half2 ds; int8_t qs[QK4_0]; };        // 36B

__device__ __forceinline__ int dp4a_sw(int a,int b,int c){
  int s=c; s+=((a<<24)>>24)*((b<<24)>>24); s+=((a<<16)>>24)*((b<<16)>>24);
  s+=((a<<8)>>24)*((b<<8)>>24); s+=(a>>24)*(b>>24); return s;
}
__device__ __forceinline__ int gi_u8(const uint8_t*x8,int i){
  const uint16_t*x16=(const uint16_t*)(x8+4*i); int x=0; x|=x16[0]; x|=x16[1]<<16; return x;
}
__device__ __forceinline__ int gi_i8a(const int8_t*x8,int i){ return *((const int*)(x8+4*i)); }

// Faithful vec_dot for Q4_0/Q8_1, vdr=2 (mirrors vecdotq.cuh)
__device__ __forceinline__ float vdot_q4_0(const block_q4_0*b4,const block_q8_1*b8,int iqs){
  int v[2],u[4];
  #pragma unroll
  for(int i=0;i<2;i++){ v[i]=gi_u8(b4->qs,iqs+i);
    u[2*i+0]=gi_i8a(b8->qs,iqs+i); u[2*i+1]=gi_i8a(b8->qs,iqs+i+(QK4_0/8)); }
  int sumi=0; const int qr=8;
  #pragma unroll
  for(int i=0;i<2;i++){ int vi0=(v[i]>>0)&0x0F0F0F0F; int vi1=(v[i]>>4)&0x0F0F0F0F;
    sumi=dp4a_sw(vi0,u[2*i+0],sumi); sumi=dp4a_sw(vi1,u[2*i+1],sumi); }
  float d8=__half2float(b8->ds.x), d4=__half2float(b4->d);
  return d4*d8*(sumi - qr* /*offset approx*/ 0); // offset folded out for bw test
}

// AoS kernel (template MMV_Y)
template<int MY>
__global__ void mmvq_aos(const block_q4_0*__restrict__ x,const block_q8_1*__restrict__ y,
                         float*__restrict__ dst,int ncols,int nrows){
  int row=blockIdx.x*MY+threadIdx.y; if(row>=nrows)return;
  int bpr=ncols/QK4_0; const int qi=QK4_0/8; const int bpw=2*WARP/qi;
  float tmp=0.f;
  for(int i=threadIdx.x/(qi/2); i<bpr; i+=bpw){
    int iqs=2*(threadIdx.x%(qi/2));
    tmp+=vdot_q4_0(&x[row*bpr+i],&y[i],iqs);
  }
  #pragma unroll
  for(int m=WARP/2;m>0;m>>=1) tmp+=__shfl_xor_sync(0xffffffff,tmp,m);
  if(threadIdx.x==0) dst[row]=tmp;
}

// SoA kernel: 128-bit quant loads + separate scale, one thread per block
template<int MY>
__global__ void mmvq_soa(const uint4*__restrict__ q,const half*__restrict__ sc,
                         const block_q8_1*__restrict__ y,float*__restrict__ dst,int ncols,int nrows){
  int row=blockIdx.x*MY+threadIdx.y; if(row>=nrows)return;
  int bpr=ncols/QK4_0; float tmp=0.f;
  for(int i=threadIdx.x;i<bpr;i+=WARP){
    uint4 qq=q[row*bpr+i];                 // 16B aligned coalesced
    const block_q8_1*b8=&y[i];
    const int*vi=(const int*)&qq;
    int sumi=0;
    #pragma unroll
    for(int k=0;k<4;k++){ int v0=(vi[k]>>0)&0x0F0F0F0F; int v1=(vi[k]>>4)&0x0F0F0F0F;
      sumi=dp4a_sw(v0,gi_i8a(b8->qs,k),sumi); sumi=dp4a_sw(v1,gi_i8a(b8->qs,k+4),sumi); }
    float d8=__half2float(b8->ds.x),d4=__half2float(sc[row*bpr+i]);
    tmp+=d4*d8*sumi;
  }
  #pragma unroll
  for(int m=WARP/2;m>0;m>>=1) tmp+=__shfl_xor_sync(0xffffffff,tmp,m);
  if(threadIdx.x==0) dst[row]=tmp;
}

// SoA but only 4-byte-aligned int loads (no uint4) — tests if contiguity alone
// (same-shape within-row reorder) captures the win without 16B alignment.
// Layout per row: [16*bpr quant bytes contiguous][2*bpr scale bytes]
template<int MY>
__global__ void mmvq_soa4(const uint8_t*__restrict__ base,const block_q8_1*__restrict__ y,
                          float*__restrict__ dst,int ncols,int nrows){
  int row=blockIdx.x*MY+threadIdx.y; if(row>=nrows)return;
  int bpr=ncols/QK4_0;
  const uint8_t* rowq=base+(size_t)row*(bpr*18);   // same 18B/row stride, reordered within
  const int* q=(const int*)rowq;                    // quants: 4 ints per block, contiguous
  const half* sc=(const half*)(rowq+bpr*16);        // scales after quants
  float tmp=0.f;
  for(int i=threadIdx.x;i<bpr;i+=WARP){
    const int* qb=q+i*4;                             // 16 bytes for block i
    const block_q8_1*b8=&y[i];
    int sumi=0;
    #pragma unroll
    for(int k=0;k<4;k++){ int v0=(qb[k]>>0)&0x0F0F0F0F; int v1=(qb[k]>>4)&0x0F0F0F0F;
      sumi=dp4a_sw(v0,gi_i8a(b8->qs,k),sumi); sumi=dp4a_sw(v1,gi_i8a(b8->qs,k+4),sumi); }
    float d8=__half2float(b8->ds.x),d4=__half2float(sc[i]);
    tmp+=d4*d8*sumi;
  }
  #pragma unroll
  for(int m=WARP/2;m>0;m>>=1) tmp+=__shfl_xor_sync(0xffffffff,tmp,m);
  if(threadIdx.x==0) dst[row]=tmp;
}

template<int MY> float timeK(void(*launch)(dim3,dim3),int nrows){return 0;}

int main(){
  int nrows=16384,ncols=4096; int bpr=ncols/QK4_0; size_t nb=(size_t)nrows*bpr;
  size_t aos=nb*18, qb=nb*16, sb=nb*2, yb=(size_t)bpr*36;
  block_q4_0*d_aos; cudaMalloc(&d_aos,aos);
  uint4*d_q; cudaMalloc(&d_q,qb); half*d_s; cudaMalloc(&d_s,sb);
  block_q8_1*d_y; cudaMalloc(&d_y,yb); float*d_dst; cudaMalloc(&d_dst,nrows*4);
  cudaMemset(d_aos,1,aos);cudaMemset(d_q,1,qb);cudaMemset(d_s,1,sb);cudaMemset(d_y,1,yb);
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);int IT=300;
  #define BENCH(NAME,MY,CALL,BYTES) do{ \
    dim3 blk(WARP,MY,1),grd((nrows+MY-1)/MY,1,1); \
    for(int i=0;i<10;i++){CALL;} cudaDeviceSynchronize(); \
    cudaEventRecord(a); for(int i=0;i<IT;i++){CALL;} cudaEventRecord(b);cudaEventSynchronize(b); \
    float ms;cudaEventElapsedTime(&ms,a,b);ms/=IT; \
    printf("%-14s MY=%d: %.3f ms, %.1f GB/s (%.0f%% of 73)\n",NAME,MY,ms,(BYTES)/(ms*1e-3)/1e9,(BYTES)/(ms*1e-3)/1e9/73*100);}while(0)
  BENCH("AoS",1,(mmvq_aos<1><<<grd,blk>>>(d_aos,d_y,d_dst,ncols,nrows)),(double)aos);
  BENCH("AoS",2,(mmvq_aos<2><<<grd,blk>>>(d_aos,d_y,d_dst,ncols,nrows)),(double)aos);
  BENCH("AoS",4,(mmvq_aos<4><<<grd,blk>>>(d_aos,d_y,d_dst,ncols,nrows)),(double)aos);
  BENCH("SoA",1,(mmvq_soa<1><<<grd,blk>>>(d_q,d_s,d_y,d_dst,ncols,nrows)),(double)(qb+sb));
  BENCH("SoA",2,(mmvq_soa<2><<<grd,blk>>>(d_q,d_s,d_y,d_dst,ncols,nrows)),(double)(qb+sb));
  BENCH("SoA",4,(mmvq_soa<4><<<grd,blk>>>(d_q,d_s,d_y,d_dst,ncols,nrows)),(double)(qb+sb));
  // SoA4: within-row reorder, same 18B/row, 4-byte int loads (no uint4)
  uint8_t* d_b; cudaMalloc(&d_b,aos); cudaMemset(d_b,1,aos);
  BENCH("SoA4(reorder)",1,(mmvq_soa4<1><<<grd,blk>>>(d_b,d_y,d_dst,ncols,nrows)),(double)aos);
  BENCH("SoA4(reorder)",2,(mmvq_soa4<2><<<grd,blk>>>(d_b,d_y,d_dst,ncols,nrows)),(double)aos);
  BENCH("SoA4(reorder)",4,(mmvq_soa4<4><<<grd,blk>>>(d_b,d_y,d_dst,ncols,nrows)),(double)aos);
  return 0;
}

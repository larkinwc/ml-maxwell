// Q6_K matvec via fp32 dequant + FMA (no emulated dp4a). Self-contained:
// random bytes -> host double reference -> GPU kernel -> max abs err + timing.
// Build: nvcc -arch=sm_50 -O3 q6k_fma.cu -o q6k_fma
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#define QK_K 256
#define WARP 32

struct __align__(2) block_q6_K {
  uint8_t ql[QK_K/2];      // 128
  uint8_t qh[QK_K/4];      // 64
  int8_t  scales[QK_K/16]; // 16
  half    d;               // 2 => 210 bytes
};

// ---- host reference: ggml dequantize_row_q6_K + dot with x ----
static void host_ref(const block_q6_K* X, const float* x, double* out, int nrows, int ncols){
  int bpr = ncols/QK_K;
  for(int r=0;r<nrows;r++){
    double acc=0.0;
    for(int b=0;b<bpr;b++){
      const block_q6_K* bq=&X[r*bpr+b];
      float d=__half2float(bq->d);
      const uint8_t* ql=bq->ql; const uint8_t* qh=bq->qh; const int8_t* sc=bq->scales;
      // 2 groups of 128, std ggml layout
      for(int n=0;n<QK_K;n+=128){
        const uint8_t* QL=ql + (n/128)*64;
        const uint8_t* QH=qh + (n/128)*32;
        const int8_t*  SC=sc + (n/128)*8;
        for(int l=0;l<32;l++){
          int is=l/16;
          int q1=(int)((QL[l]   & 0xF)|(((QH[l]>>0)&3)<<4)) - 32;
          int q2=(int)((QL[l+32]& 0xF)|(((QH[l]>>2)&3)<<4)) - 32;
          int q3=(int)((QL[l]   >> 4)|(((QH[l]>>4)&3)<<4)) - 32;
          int q4=(int)((QL[l+32]>> 4)|(((QH[l]>>6)&3)<<4)) - 32;
          int col=b*256+n;
          acc += (double)(d*SC[is+0]*q1) * x[col+l+0];
          acc += (double)(d*SC[is+2]*q2) * x[col+l+32];
          acc += (double)(d*SC[is+4]*q3) * x[col+l+64];
          acc += (double)(d*SC[is+6]*q4) * x[col+l+96];
        }
      }
    }
    out[r]=acc;
  }
}

// ---- GPU kernel: warp per row (MY rows/block), thread=l (0..31) ----
template<int MY>
__global__ void mv(const block_q6_K* __restrict__ X, const half* __restrict__ x,
                   float* __restrict__ dst, int ncols, int nrows){
  int row=blockIdx.x*MY+threadIdx.y; if(row>=nrows)return;
  int bpr=ncols/QK_K; int l=threadIdx.x; float acc=0.f;
  for(int b=0;b<bpr;b++){
    const block_q6_K* bq=&X[row*bpr+b];
    float d=__half2float(bq->d);
    const uint8_t* ql=bq->ql; const uint8_t* qh=bq->qh; const int8_t* sc=bq->scales;
    int base=b*256;
    #pragma unroll
    for(int g=0;g<2;g++){
      const uint8_t* QL=ql+g*64; const uint8_t* QH=qh+g*32; const int8_t* SC=sc+g*8;
      int n=g*128; int is=l/16;
      uint8_t ql_l=QL[l], ql_l32=QL[l+32], qh_l=QH[l];
      int q1=(int)((ql_l   &0xF)|(((qh_l>>0)&3)<<4))-32;
      int q2=(int)((ql_l32 &0xF)|(((qh_l>>2)&3)<<4))-32;
      int q3=(int)((ql_l   >>4)|(((qh_l>>4)&3)<<4))-32;
      int q4=(int)((ql_l32 >>4)|(((qh_l>>6)&3)<<4))-32;
      acc += d*SC[is+0]*q1*__half2float(x[base+n+l+0]);
      acc += d*SC[is+2]*q2*__half2float(x[base+n+l+32]);
      acc += d*SC[is+4]*q3*__half2float(x[base+n+l+64]);
      acc += d*SC[is+6]*q4*__half2float(x[base+n+l+96]);
    }
  }
  #pragma unroll
  for(int m=WARP/2;m>0;m>>=1) acc+=__shfl_xor_sync(0xffffffff,acc,m);
  if(l==0) dst[row]=acc;
}

int main(){
  int nrows=8192, ncols=4096;     // chunk; scale timing to 834MB lm_head separately
  int bpr=ncols/QK_K; size_t nb=(size_t)nrows*bpr;
  block_q6_K* hX=(block_q6_K*)malloc(nb*sizeof(block_q6_K));
  for(size_t i=0;i<nb*sizeof(block_q6_K);i++) ((uint8_t*)hX)[i]=rand()&0xff;
  // set d to small positive halfs to keep magnitudes sane
  for(size_t b=0;b<nb;b++) hX[b].d=__float2half(0.01f);
  float* hx=(float*)malloc(ncols*4); half* hxh=(half*)malloc(ncols*2);
  for(int c=0;c<ncols;c++){ hx[c]=((rand()%200)-100)/100.f; hxh[c]=__float2half(hx[c]); }
  double* href=(double*)malloc(nrows*8); host_ref(hX,hx,href,nrows,ncols);

  block_q6_K* dX; half* dx; float* dst;
  cudaMalloc(&dX,nb*sizeof(block_q6_K)); cudaMalloc(&dx,ncols*2); cudaMalloc(&dst,nrows*4);
  cudaMemcpy(dX,hX,nb*sizeof(block_q6_K),cudaMemcpyHostToDevice);
  cudaMemcpy(dx,hxh,ncols*2,cudaMemcpyHostToDevice);

  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); int IT=300;
  size_t wbytes=nb*sizeof(block_q6_K);
  float* hdst=(float*)malloc(nrows*4);
  #define RUN(MY) do{ \
    dim3 blk(WARP,MY,1),grd((nrows+MY-1)/MY,1,1); \
    for(int i=0;i<10;i++) mv<MY><<<grd,blk>>>(dX,dx,dst,ncols,nrows); \
    cudaError_t e=cudaDeviceSynchronize(); if(e){printf("MY=%d ERR %s\n",MY,cudaGetErrorString(e));break;} \
    cudaMemcpy(hdst,dst,nrows*4,cudaMemcpyDeviceToHost); \
    double maxrel=0; for(int r=0;r<nrows;r++){double ref=href[r];double err=fabs(hdst[r]-ref);double den=fabs(ref)+1e-3; if(err/den>maxrel)maxrel=err/den;} \
    cudaEventRecord(a); for(int i=0;i<IT;i++) mv<MY><<<grd,blk>>>(dX,dx,dst,ncols,nrows); cudaEventRecord(b);cudaEventSynchronize(b); \
    float ms;cudaEventElapsedTime(&ms,a,b);ms/=IT; \
    printf("fp32-FMA MY=%d: %.4f ms, %.1f GB/s (%.0f%% of 73)  maxrel=%.2e\n",MY,ms,wbytes/(ms*1e-3)/1e9,wbytes/(ms*1e-3)/1e9/73*100,maxrel);}while(0)
  RUN(1); RUN(2); RUN(4); RUN(8);
  printf("(baseline real op Q6_K = 33 GB/s / 46%%)\n");
  return 0;
}



#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cassert>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do {                              \
  cudaError_t err = (call);                                 \
  if (err != cudaSuccess) {                                 \
    fprintf(stderr, "CUDA error %s:%d: %s\n",               \
            __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(1);                                                \
  }                                                         \
} while(0)
#endif

// -------- Utilities ----------
static inline float randf() { return (float)rand()/RAND_MAX - 0.5f; }
static inline float dot(const float* a, const float* b, int d) {
  float s=0.f;
  for (int i=0;i<d;++i) s += a[i]*b[i];
  return s;
}
__host__ __device__ static inline float maxf(float a, float b){ return a>b?a:b; }


void attention_naive_cpu(const float* Q, const float* K, const float* V,
                         float* O, int N, int d, bool causal)
{
  const float scale = 1.f / std::sqrt((float)d);
  
  float* S = (float*)malloc(sizeof(float)*N);
  float* P = (float*)malloc(sizeof(float)*N);

  for (int i=0;i<N;++i){
    float m = -INFINITY;
    for (int j=0;j<N;++j){
      float s = dot(Q + i*d, K + j*d, d) * scale;
      if (causal && j>i) s = -INFINITY;
      S[j] = s;
      m = maxf(m, s);
    }
    float denom = 0.f;
    for (int j=0;j<N;++j){
      float p = (S[j]==-INFINITY)? 0.f : expf(S[j]-m);
      P[j] = p;
      denom += p;
    }
    for (int k=0;k<d;++k) O[i*d+k]=0.f;
    float inv = 1.f / denom;
    for (int j=0;j<N;++j){
      float w = P[j]*inv;
      const float* Vj = V + j*d;
      for (int k=0;k<d;++k) O[i*d+k] += w * Vj[k];
    }
  }
  free(S); free(P);
}


void flash_attention_cpu(const float* Q, const float* K, const float* V,
                         float* O, int N, int d, bool causal,
                         int BLOCK_N = 64) // tile width on K/V
{
  const float scale = 1.f / std::sqrt((float)d);
  float* acc_t = (float*)malloc(sizeof(float)*d);
  float* s_buf = (float*)malloc(sizeof(float)*BLOCK_N);

  for (int i=0;i<N;++i){
    float m = -INFINITY;     // running max
    float l = 0.f;           // running denominator

    float* acc = (float*)calloc(d, sizeof(float));

    for (int j0=0; j0<N; j0+=BLOCK_N){
      int bn = std::min(BLOCK_N, N - j0);

      float mt = -INFINITY;
      for (int jj=0; jj<bn; ++jj){
        int j = j0 + jj;
        float s = dot(Q + i*d, K + j*d, d) * scale;
        if (causal && j>i) s = -INFINITY;
        s_buf[jj] = s;
        mt = maxf(mt, s);
      }

      float lt = 0.f;
      for (int k=0;k<d;++k) acc_t[k]=0.f;

      for (int jj=0; jj<bn; ++jj){
        float s = s_buf[jj];
        if (s== -INFINITY) continue;
        float p = expf(s - mt);
        lt += p;
        const float* Vj = V + (j0+jj)*d;
        for (int k=0;k<d;++k) acc_t[k] += p * Vj[k];
      }

      float m_new = maxf(m, mt);
      float alpha = (m == -INFINITY)? 0.f : expf(m  - m_new);
      float beta  = (mt== -INFINITY)? 0.f : expf(mt - m_new);

      l = l*alpha + lt*beta;
      for (int k=0;k<d;++k) acc[k] = acc[k]*alpha + acc_t[k]*beta;
      m = m_new;
    }

    float inv = 1.f / l;
    for (int k=0;k<d;++k) O[i*d+k] = acc[k] * inv;
    free(acc);
  }

  free(acc_t);
  free(s_buf);
}


template<int ROWS_PER_BLOCK, int BLOCK_N>
__global__ void flash_attn_kernel_smemacc(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ O,
                                          int N, int d, bool causal)
{
  extern __shared__ float smem[];
  float* Q_s = smem;
  float* K_s = Q_s + ROWS_PER_BLOCK * d;
  float* V_s = K_s + BLOCK_N * d;
  float* ACC = V_s + BLOCK_N * d;

  const int row_in_block = threadIdx.y;
  const int lane         = threadIdx.x;
  const int i = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
  if (i >= N) return;

  const float scale = 1.f / sqrtf((float)d);

  // load Q block
  for (int t = lane; t < d; t += blockDim.x) {
    Q_s[row_in_block*d + t] = Q[i*d + t];
    ACC[row_in_block*d + t] = 0.f;
  }
  __syncthreads();

  float m = -INFINITY; // running max
  float l = 0.f;       // running denom

  for (int j0=0; j0<N; j0+=BLOCK_N) {
    int bn = min(BLOCK_N, N - j0);

    // load K/V tile
    int tile_elems = bn * d;
    for (int t = lane; t < tile_elems; t += blockDim.x) {
      int r = t / d, c = t % d;
      K_s[r*d + c] = K[(j0 + r)*d + c];
      V_s[r*d + c] = V[(j0 + r)*d + c];
    }
    __syncthreads();

    // PASS-1: tile max
    float mt = -INFINITY;
    for (int jj=0; jj<bn; ++jj){
      const float* Qi = Q_s + row_in_block*d;
      const float* Kj = K_s + jj*d;
      float s = 0.f;
      for (int c=0;c<d;++c) s += Qi[c]*Kj[c];
      s *= scale;
      if (causal && (j0+jj)>i) s = -INFINITY;
      mt = maxf(mt, s);
    }

    float m_new = maxf(m, mt);
    float alpha = (m == -INFINITY)? 0.f : expf(m  - m_new);
    float beta  = (mt== -INFINITY)? 0.f : expf(mt - m_new);

    // ACC = ACC * alpha
    for (int c = lane; c < d; c += blockDim.x) {
      ACC[row_in_block*d + c] *= alpha;
    }
    __syncthreads();

    float lt = 0.f;
    for (int jj=0; jj<bn; ++jj){
      const float* Qi = Q_s + row_in_block*d;
      const float* Kj = K_s + jj*d;
      float s = 0.f;
      for (int c=0;c<d;++c) s += Qi[c]*Kj[c];
      s *= scale;
      if (causal && (j0+jj)>i) continue;
      float p = expf(s - mt);
      lt += p;
      const float* Vj = V_s + jj*d;
      for (int c = lane; c < d; c += blockDim.x) {
        atomicAdd(&ACC[row_in_block*d + c], beta * p * Vj[c]);
      }
      __syncthreads();
    }

    l = l*alpha + lt*beta;
    m = m_new;
  }

  float inv = 1.f / l;
  for (int c = lane; c < d; c += blockDim.x) {
    O[i*d + c] = ACC[row_in_block*d + c] * inv;
  }
}

// Host wrapper
void flash_attention_cuda(const float* Q, const float* K, const float* V,
                          float* O, int N, int d, bool causal,
                          int rows_per_block = 32, int block_n = 64)
{
  dim3 block(32, 32); // 32 lanes × 32 rows = 1024 threads/block
  if (rows_per_block != 32 || block_n != 64) {
    fprintf(stderr, "For simplicity, this demo expects ROWS_PER_BLOCK=32,BLOCK_N=64.\n");
    exit(1);
  }
  dim3 grid((N + rows_per_block - 1)/rows_per_block);

  size_t smem = (rows_per_block + 2*block_n + rows_per_block) * (size_t)d * sizeof(float);
  // = Q_s + K_s + V_s + ACC

  flash_attn_kernel_smemacc<32,64><<<grid, block, smem>>>(
      Q, K, V, O, N, d, causal);
  CHECK_CUDA(cudaGetLastError());
}

int main(){
  srand(0);
  const int N = 256;       // sequence length
  const int d = 64;        // hidden dim
  const bool causal = true;

  size_t bytes = (size_t)N*d*sizeof(float);
  float *hQ=(float*)malloc(bytes), *hK=(float*)malloc(bytes), *hV=(float*)malloc(bytes);
  float *hO0=(float*)malloc(bytes), *hO1=(float*)malloc(bytes), *hO2=(float*)malloc(bytes);

  for (int i=0;i<N*d;++i){ hQ[i]=randf(); hK[i]=randf(); hV[i]=randf(); }

  // CPU naive
  attention_naive_cpu(hQ,hK,hV,hO0,N,d,causal);

  // CPU flash 
  flash_attention_cpu(hQ,hK,hV,hO1,N,d,causal,64);

  // CUDA
  float *dQ,*dK,*dV,*dO;
  CHECK_CUDA(cudaMalloc(&dQ, bytes));
  CHECK_CUDA(cudaMalloc(&dK, bytes));
  CHECK_CUDA(cudaMalloc(&dV, bytes));
  CHECK_CUDA(cudaMalloc(&dO, bytes));
  CHECK_CUDA(cudaMemcpy(dQ,hQ,bytes,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK,hK,bytes,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV,hV,bytes,cudaMemcpyHostToDevice));

  flash_attention_cuda(dQ,dK,dV,dO,N,d,causal,32,64);
  CHECK_CUDA(cudaMemcpy(hO2,dO,bytes,cudaMemcpyDeviceToHost));

  // Compare
  auto max_abs_diff = [&](const float* A, const float* B){
    double m=0; for (int i=0;i<N*d;++i) m = std::max(m, (double)fabs(A[i]-B[i]));
    return m;
  };
  double diff_cpu = max_abs_diff(hO0,hO1);
  double diff_gpu = max_abs_diff(hO0,hO2);
  printf("Max|CPU-naive - CPU-flash| = %.6g\n", diff_cpu);
  printf("Max|CPU-naive - CUDA-flash|= %.6g\n", diff_gpu);

  CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK));
  CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
  free(hQ); free(hK); free(hV); free(hO0); free(hO1); free(hO2);
  return 0;
}

/*
Summary:
  求softmax

Details:
  求最大值和求和两次reduce规约

缺点:
  用了一个block做计算，bs > 1时无法work
*/
#include <float.h>
#include "stdio.h"
#include "stdlib.h"

#define N 10240
#define BLOCK_NUM 1
#define THREAD_NUM 1024
#define MAX(a, b) (a > b ? a : b)


__device__ float reduce_max(float val) {
  for (int offset = (warpSize >> 1); offset >= 1; offset >>= 1) {
    val = MAX(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ float reduce_sum(float val) {
  for (int offset = (warpSize >> 1); offset >= 1; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void softmax(float* d_in, float* d_out, int n) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float mx = -FLT_MAX;

  for (int i = global_idx; i < n; i += stride) {
    mx = MAX(mx, d_in[i]);
  }
  mx = reduce_max(mx);

  __shared__ float buf_max[32];

  int wid = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;

  if (lane == 0) {
    buf_max[wid] = mx;
  }
  
  __syncthreads();

  if (wid == 0) {
    int num_warps = blockDim.x / 32;
    float val = (lane < num_warps) ? buf_max[lane] : -FLT_MAX;
    mx = reduce_max(val);
    if (lane == 0) {
      buf_max[0] = mx;
    }
  }
  __syncthreads();

  float global_max = buf_max[0];

  float sum = 0;
  for (int i = global_idx; i < n; i += stride) {
    d_in[i] = expf(d_in[i] - global_max);
    sum += d_in[i];
  }

  sum = reduce_sum(sum);

  __shared__ float buf_sum[32];

  if (lane == 0) {
    buf_sum[wid] = sum;
  }

  __syncthreads();

  if (wid == 0) {
    int num_warps = blockDim.x / 32;
    float val = (lane < num_warps) ? buf_sum[lane] : 0.0f;
    sum = reduce_sum(sum);
    if (lane == 0) {
      buf_sum[0] = sum;
    }
  }

  __syncthreads();
  
  float global_sum = buf_sum[0];

  for (int i = global_idx; i < n; i += stride) {
    d_out[i] = d_in[i] / global_sum;
  }
}


int main() {
  int size = sizeof(float) * N;
  float *in = (float*)malloc(size);
  float *out = (float*)malloc(size);

  for (int i = 0; i < N; ++i) {
    in[i] = 1;
  }

  float *d_in, *d_out;
  cudaMalloc((void**)&d_in, size);
  cudaMalloc((void**)&d_out, size);

  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);
  softmax<<<BLOCK_NUM, THREAD_NUM>>>(d_in, d_out, N);

  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i) {
    printf("i=%d, softmax_value=%f \n", i, out[i]);
  }
  free(in);
  free(out);
  cudaFree(d_in);
  cudaFree(d_out);
}

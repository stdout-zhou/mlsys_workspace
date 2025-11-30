/*
Summary:
  对数组求和

Details:
  在v2版本的基础上，增加了两种优化。
  * warp折半规约时的循环进行展开。
  * 向量化加载，每次可以从memory load 4个int到寄存器。
*/

#include <stdio.h>
#include <stdlib.h>


#define N 10240

#define THREADS_PER_BLOCK 1024

__device__ int res = 0;

__device__ int warp_reduce_sum(int sum) {
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

__global__ void vector_sum(int* arr, int n) {
  int tid = threadIdx.x;
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int sum = 0;
  int stride = gridDim.x * blockDim.x;

  int4* arr4 = (int4*)arr;
  int n4 = n / 4;
  for (int i = global_idx; i < n4; i += stride) {
    int4 v = arr4[i];
    sum += v.x + v.y + v.z + v.w;
  }
  for (int i = n4 * 4 + global_idx; i < n; i += stride) {
      sum += arr[i];
  }

  int wid = tid / warpSize;
  int lane = tid % warpSize;
  sum = warp_reduce_sum(sum);
  __shared__ int buf[32];
  if (lane == 0) {
    buf[wid] = sum;
  }
  __syncthreads();

  sum = (tid < 32) ? buf[lane] : 0;
  if (wid == 0) {
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      atomicAdd(&res, sum);
    }
  }
}


int main() {
  int size = N * sizeof(int);
  int* arr = (int*)malloc(size);
  for (int i = 0; i < N; ++i) {
    arr[i] = 2;
  }
  int* d_arr;
  cudaMalloc((void**)&d_arr, size);
  cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);
  
  int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  vector_sum<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_arr, N);
  cudaDeviceSynchronize();

  int h_res;
  cudaMemcpyFromSymbol(&h_res, res, sizeof(int), 0, cudaMemcpyDeviceToHost);
  printf("res valus is %d\n", h_res);

  free(arr);
  cudaFree(d_arr);
}

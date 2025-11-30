/*
Summary:
  对数组求和

Details:
  每个warp里进行折半规约，用`__shfl_down_sync`做warp的同步
  比block里的barrier要快，然后使用`atomicAdd`进行原子加法

缺点:
  老问题，每次规约时，至少一半的线程计算对结果没有影响。

  shared_memory的大小分配是静态的，编译期就要确定好。
*/

#include <stdio.h>
#include <stdlib.h>


#define N 10240

#define THREADS_PER_BLOCK 1024

__device__ int res = 0;

__device__ int warp_reduce_sum(int sum) {
  for (int offset = (warpSize >> 1); offset >= 1; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  return sum;
}

__global__ void vector_sum(int* arr, int n) {
  int tid = threadIdx.x;
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int sum = 0;
  int stride = gridDim.x * blockDim.x;
  for (int i = global_idx; i < n; i += stride) {
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

/*
Summary:
  对数组求和

Details:
  每个block里进行折半规约，然后使用`atomicAdd`进行原子加法

缺点:
  每个block间规约时都设置了barrier，但实际上cuda是按照warp来调度线程的，
  warp之间不需要barrier就可以同步。

  折半规约的一个问题是每次规约时，至少一半的线程计算对结果没有影响。
*/
#include <stdio.h>
#include <stdlib.h>


#define N 10240

#define THREADS_PER_BLOCK 1024

__device__ int res = 0;

__global__ void vector_sum(int* arr, int n) {
  int tid = threadIdx.x;
  int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ int buf[THREADS_PER_BLOCK];
  if (global_idx < n) {
    buf[tid] = arr[global_idx];
  } else {
    buf[tid] = 0;
  }
  __syncthreads();
  for (int i = (THREADS_PER_BLOCK >> 1); i >= 1; i >>= 1) {
    if (tid < i) {
      buf[tid] += buf[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(&res, buf[0]);
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

/*
Summary:
矩阵乘法
A: N * K 
B: K * M
C: N * M

Detai:
最朴素的实现，每个线程负责计算输出矩阵中的一个位置
*/

#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define M 1024
#define K 1024

#define BLOCK_NUM 32


__global__ void gemm(int *a, int *b, int *c) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < N && col < M) {
    int sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += a[row * K + k] * b[k * M + col];
    }
    c[row * M + col] = sum;
  }
}

int main() {
  int size_a = N * K * sizeof(int);
  int size_b = K * M * sizeof(int);
  int size_c = N * M * sizeof(int);

  int *a = (int*)malloc(size_a);
  int *b = (int*)malloc(size_b);
  int *c = (int*)malloc(size_c);
  
  for (int i = 0; i < size_a / sizeof(int); ++i) {
    a[i] = 1;
  }
  for (int i = 0; i < size_b / sizeof(int); ++i) {
    b[i] = 1;
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, size_a);
  cudaMalloc((void**)&d_b, size_b);
  cudaMalloc((void**)&d_c, size_c);

  cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, size_a, cudaMemcpyHostToDevice);

  gemm<<<dim3(BLOCK_NUM, BLOCK_NUM), dim3(N / BLOCK_NUM, M / BLOCK_NUM)>>>(d_a, d_b, d_c);

  cudaDeviceSynchronize();

  cudaMemcpy(c, d_c, size_a, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; ++i) {
    printf("value=%d\n", c[i]);
  }

  free(a);
  free(b);
  free(c);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

/*
Summary:
矩阵乘法
A: N * K 
B: K * M
C: N * M

Detail:
在v1版本上增加shared memory优化

Motivation:
计算C的每个位置都要读A的一整行和B的一整列，但是这里有很多重复读取，
比如我只读A的一行，然后循环读B的列，很明显这样能算出C的一整行。

考虑一种更优的读取策略。
矩阵的乘法是可以分块计算的，更形象的来说，一行和一列求乘和时，可以把行列
分成若干个tile，每个tile求乘和，最后求和，得到的结果不变。
如果把tile看成是一个二维的结构(32 * 32), 那么一个长度为32的tile_row, 可以
和32个tile_col运算，如果用share memory后，只有1/32的load global memory次数。

具体实现时，可以看成一个方块在A矩阵上沿着行滑动，在B矩阵上沿着列滑动。
可以得到实现:
__global__ void gemm(int *a, int *b, int *c) {
  __shared__ float a_buf[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float b_buf[BLOCK_SIZE][BLOCK_SIZE];
  int sum = 0;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.x * BLOCK_SIZE + tx;
  int col = blockIdx.y * BLOCK_SIZE + ty;

  for (int i = 0; i < K / BLOCK_SIZE; ++i) {
    a_buf[tx][ty] = a[row * K + i * BLOCK_SIZE + ty];
    b_buf[tx][ty] = b[(i * BLOCK_SIZE + tx) * M + col];

    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; ++i) {
      sum += a_buf[tx][i] * b_buf[i][ty];
    }
    __syncthreads();
  }
  c[row * M + col] = sum;
}

这种实现性能是有问题的
a_buf[tx][ty] = a[row * K + i * BLOCK_SIZE + ty];
众所周知，cuda是按照warp调度线程的，但是上面的这句代码在一个warp内
的访存确并不连续，如果能优化成连续访存，gpu会把warp内32个thread合并成
一次访存。

此外
sum += a_buf[tx][i] * b_buf[i][ty];
在一个warp内，这句代码的i一样，但tx不一样，这意味着从shared memory
访存时，会访问同一个bank，这里会有bank conflict
所以在实现上，需要颠倒row, col, 而这除了改变访存顺序外并不影响结果
*/

#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define M 1024
#define K 1024

#define BLOCK_NUM 32

#define BLOCK_SIZE 32


__global__ void gemm(int *a, int *b, int *c) {
  __shared__ float a_buf[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float b_buf[BLOCK_SIZE][BLOCK_SIZE];
  int sum = 0;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;

  for (int i = 0; i < K / BLOCK_SIZE; ++i) {
    a_buf[ty][tx] = (float)a[row * K + (i * BLOCK_SIZE + tx)];
    b_buf[ty][tx] = (float)b[(i * BLOCK_SIZE + ty) * M + col];

    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; ++i) {
      sum += a_buf[ty][i] * b_buf[i][tx];
    }
    __syncthreads();
  }
  c[row * M + col] = sum;
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

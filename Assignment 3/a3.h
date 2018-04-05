#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define THREAD_COUNT 16
#define BLOCK_DIM_X  32
#define BLOCK_DIM_Y  32

__global__ void
gen_i(float *dA, uint16_t N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        dA[i + N * i] = 1;
    }
}

__global__ void
scale_submat(float *dA, float alpha, uint16_t N, uint16_t start_i, uint16_t start_j) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i > start_i && i < N && j > start_j && j < N) {
        dA[i + N * j] = alpha * dA[i + N * j];
    }
}

__global__ void reduce0(float *g_idata0, float *g_idata1, float *g_odata, uint16_t n) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("tid: %u\ti: %u\n", tid, i);

    //printf("i: %u, n: %u\n", i, n);
    //sdata[tid] = abs(g_idata0[i] - g_idata1[i]);
    sdata[tid] = (i < n) ? fabsf(g_idata0[i] - g_idata1[i]) : 0;
    //printf("%f\n", sdata[tid]);
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void
reduce6(float *g_idata0, float *g_idata1, float *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += fabsf((g_idata0[i] + g_idata0[i+blockSize]) -
                      (g_idata1[i] + g_idata1[i+blockSize]));
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) {
            if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
            if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
            if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
            if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
            if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
            if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
        }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

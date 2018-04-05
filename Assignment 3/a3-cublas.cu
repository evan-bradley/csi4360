#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

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

void gen_identity (float *A, uint16_t N) {
  for (uint16_t i = 0; i < N; i++) {
    A[i + N * i] = 1.0;
  }
}

void gen_x (float *A, uint16_t N) {
  for (uint16_t i = 0; i < N; i++) {
    for (uint16_t j = 0; j < N; j++) {
      A[i + N * j] = rand() / (float) RAND_MAX;
    }
  }
}

void print_mat (float *A, uint16_t N) {
  for (uint16_t i = 0; i < N; i++) {
    for (uint16_t j = 0; j < N; j++) {
      printf("%.4f\t", A[i + N * j]);
    }
    printf("\n");
  }
}

void mm (float *A, float *B, float *C, uint16_t N) {
  for (uint16_t i = 0; i < N; i++)
    for (uint16_t j = 0; j < N; j++)
      for (uint16_t k = 0; k < N; k++)
        C[i + N * j] += A[i + N * k] * B[k + N * j];
}

void const_mult (float *A, int8_t c, uint16_t start_i, uint16_t start_j, uint16_t end) {
  for (uint16_t i = start_i; i < end; i++) {
    for (uint16_t j = start_j; j < end; j++) {
      A[i + end * j] = c * A[i + end * j];
    }
  }
}

void copy_submat (float *big, uint16_t bigN, float *small, uint16_t start_i, uint16_t start_j, uint16_t smallN) {
  for (uint16_t i = 0; i < smallN; i++) {
    for (uint16_t j = 0; j < smallN; j++) {
      big[(i + start_i) + bigN * (j + start_j)] = small[i + smallN * j];
    }
  }
}

int mm_eq (float *A, float *B, uint16_t N) {
  float error = 0.0;

  for (uint16_t i = 0; i < N; i++) {
    for (uint16_t j = 0; j < N; j++) {
      error += abs(A[i + N * j] - B[i + N * j]);
    }
  }

  return error;
}

int main(int argc, char **argv)
{

    dim3 blockDim1(BLOCK_DIM_X, 1);  
    cublasOperation_t trans = CUBLAS_OP_N;

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, 100); //(unsigned long long) clock()

    float alpha = 1.0f;
    float beta = 0.0f;

    float *dA;
    float *dB;
    float *dC;
    float *dX;

    cublasHandle_t handle = 0;

    uint16_t N = 10000;
    uint16_t N2 = 2 * N;
    dim3 gridDim1((N2 + BLOCK_DIM_X - 1) / BLOCK_DIM_X, 1);

    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK(cudaMalloc((void **)&dX, sizeof(float) * N * N));

    curandGenerateUniform(prng, dX, N * N);

    // Allocate device memory for vectors and the dense form for the matrices
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * N2 * N2));
    CHECK(cudaMemset(dA, 0, N2 * N2));
    gen_i<<<gridDim1, blockDim1>>>(dA, N2);
    CHECK_CUBLAS(cublasSetMatrix(N, N, sizeof(float), dX, N, (dA + 2 * N * N), N2));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N2 * N2));
    CHECK(cudaMemset(dA, 0, N2 * N2));
    gen_i<<<gridDim1, blockDim1>>>(dA, N2);
    alpha = -1.0f;
    CHECK_CUBLAS(cublasSgeam(handle, trans, trans, N2, N,
                 &alpha,
                 (dA + 2 * N * N), N2,
                 &beta,
                 (dA + 2 * N * N), N2,
                 (dA + 2 * N * N), N2));
    CHECK_CUBLAS(cublasSetMatrix(N, N, sizeof(float), dX, N, (dA + 2 * N * N), N2));
    alpha = 2.0f;
    CHECK_CUBLAS(cublasSgeam(handle, trans, trans, N, N,
                 &alpha,
                 (dB + 2 * N * N), N2,
                 &beta,
                 (dB + 2 * N * N), N2,
                 (dB + 2 * N *N), N2));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(dX));

    CHECK(cudaMalloc((void **)&dC, sizeof(float) * N2 * N2));
    //CHECK(cudaMalloc((void **)&dF, sizeof(float) * N2 * N2));

    CHECK_CUBLAS(cublasSgemm(handle, trans, trans, N2, N2, N2,
                             &alpha,
                             dA, N2,
                             dB, N2,
                             &beta,
                             dC, N2));

    alpha = -1.0f;
    CHECK_CUBLAS(cublasSgeam(handle, trans, trans, N, N,
                 &alpha,
                 (dA + 2 * N * N), N2,
                 &beta,
                 (dA + 2 * N * N), N2,
                 (dA + 2 * N *N), N2));


    CHECK_CUBLAS(cublasSgemm(handle, trans, trans, N2, N2, N2,
                             &alpha,
                             dC, N2,
                             dA, N2,
                             &beta,
                             dB, N2));

    // Copy the result vector back to the host
//    CHECK(cudaMemcpy(E, dE, sizeof(float) * N2 * N2, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));

    //printf("Error: %d\n", mm_eq(E, F, N2));

    //free(E);
    //free(F);

    //CHECK(cudaFree(dF));

    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}

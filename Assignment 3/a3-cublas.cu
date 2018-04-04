#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>

/*
 * M = # of rows
 * N = # of columns
 */
//int N = 1024;

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
    cublasOperation_t trans = CUBLAS_OP_N;
    //int row;

    float alpha = 1.0f;
    float beta = 1.0f;

    float* O;
    float* I;
    float* X;
    float* Y;
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    float *D, *dD;
    float *E, *dE;
    float *F;//, *dF;

    cublasHandle_t handle = 0;

    uint16_t N = 4;
    uint16_t N2 = 2 * N;

    // Generate input
    O = (float *) calloc(N * N, sizeof(float));
    I = (float *) calloc(N * N, sizeof(float));
    X = (float *) calloc(N * N, sizeof(float));
    Y = (float *) calloc(N * N, sizeof(float));
    A = (float *) calloc(N2 * N2, sizeof(float));
    B = (float *) calloc(N2 * N2, sizeof(float));
    C = (float *) calloc(N2 * N2, sizeof(float));
    D = (float *) calloc(N2 * N2, sizeof(float));
    E = (float *) calloc(N2 * N2, sizeof(float));
    F = (float *) calloc(N2 * N2, sizeof(float));

    srand(100);

    gen_x(X, N);
    copy_submat(Y, N, X, 0, 0, N);

    // [I X]
    // [O I]
    gen_identity(A, N2);
    copy_submat(A, N2, X, 0, N, N);

    // [I -X]
    // [O  I]
    gen_identity(C, N2);
    const_mult(Y, -1, 0, 0, N);
    copy_submat(C, N2, Y, 0, N, N);

    // [I 2X]
    // [O -I]
    gen_identity(B, N2);
    const_mult(B, -1, N, N, N2);
    const_mult(Y, -2, 0, 0, N);
    copy_submat(B, N2, Y, 0, N, N);

    free(O);
    free(I);
    free(X);
    free(Y);

    gen_identity(F, N2);
    const_mult(F, -1, N, N, N2);


    // A * B * C = D * C = E
    //mm(A, B, D, N2);
    //mm(D, C, E, N2);

    // Create the cuSPARSE handle
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device memory for vectors and the dense form for the matrices
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * N2 * N2));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N2 * N2));
    CHECK(cudaMalloc((void **)&dD, sizeof(float) * N2 * N2));
    //CHECK(cudaMalloc((void **)&dF, sizeof(float) * N2 * N2));

    CHECK_CUBLAS(cublasSetMatrix(N2, N2, sizeof(float), A, N2, dA, N2));
    CHECK_CUBLAS(cublasSetMatrix(N2, N2, sizeof(float), B, N2, dB, N2));
    CHECK_CUBLAS(cublasSetMatrix(N2, N2, sizeof(float), D, N2, dD, N2));
    //CHECK_CUBLAS(cublasSetMatrix(N2, N2, sizeof(float), F, N2, dF, N2));
    // Transfer the dense matrices to the device
    /*CHECK(cudaMemcpy(dA, A, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dC, C, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dD, D, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dE, E, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dF, F, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));*/

    free(A);
    free(B);
    free(D);

    CHECK_CUBLAS(cublasSgemm(handle, trans, trans, N2, N2, N2,
                             &alpha,
                             dA, N2,
                             dB, N2,
                             &beta,
                             dD, N2));

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));

    CHECK(cudaMalloc((void **)&dC, sizeof(float) * N2 * N2));
    CHECK(cudaMalloc((void **)&dE, sizeof(float) * N2 * N2));
    CHECK_CUBLAS(cublasSetMatrix(N2, N2, sizeof(float), C, N2, dC, N2));
    CHECK_CUBLAS(cublasSetMatrix(N2, N2, sizeof(float), E, N2, dE, N2));
    free(C);

    CHECK_CUBLAS(cublasSgemm(handle, trans, trans, N2, N2, N2,
                             &alpha,
                             dD, N2,
                             dC, N2,
                             &beta,
                             dE, N2));

    // Copy the result vector back to the host
    CHECK(cudaMemcpy(E, dE, sizeof(float) * N2 * N2, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dC));
    CHECK(cudaFree(dD));
    CHECK(cudaFree(dE));

    printf("Equal: %d\n", mm_eq(E, F, N2));

    free(E);
    free(F);

    //CHECK(cudaFree(dF));

    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}

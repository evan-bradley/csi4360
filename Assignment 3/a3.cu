#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cusparse_v2.h>
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

    /*if (i < N && j < N) {
        if (i == j) {
            printf("i == j == %d\n", i);
        } else {
            dA[i + N * j] = 0;
        }
        //dA[i + N * j] = i == j ? 1 : 0;
    }*/
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

void dense2csr(cusparseHandle_t handle, float *A, float *dA, cusparseMatDescr_t descr,
               int *dNnzPerRow, float *dCsrVal, int *dCsrRowPtr, int *dCsrColInd,
               int *nnz, uint16_t N)
{
    CHECK(cudaMemcpy(dA, A, sizeof(float) * N * N, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void **)dNnzPerRow, sizeof(int) * N));

    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descr, dA,
                                N, dNnzPerRow, nnz));

    CHECK(cudaMalloc((void **)dCsrVal, sizeof(float) * (*nnz)));
    CHECK(cudaMalloc((void **)dCsrRowPtr, sizeof(int) * (N + 1)));
    CHECK(cudaMalloc((void **)dCsrColInd, sizeof(int) * (*nnz)));

    CHECK_CUSPARSE(cusparseSdense2csr(handle, N, N, descr, dA, N, dNnzPerRow,
                                      dCsrVal, dCsrRowPtr, dCsrColInd));
}

int main(int argc, char **argv)
{

    cudaEvent_t start, stop;
    float elapsed;
    cudaStream_t stream_0, stream_1;

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
    dim3 blockDim1(BLOCK_DIM_X, 1);  

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, 100); //(unsigned long long) clock()

    cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cublasOperation_t trans_blas = CUBLAS_OP_N;
    cublasHandle_t handle_blas = 0;
    CHECK_CUBLAS(cublasCreate(&handle_blas));

    //int row;
    int *dNnzPerRowA,
      *dNnzPerRowB,
      *dNnzPerRowC,
      //*dNnzPerRowD,
      *dNnzPerRowE;
     // *dNnzPerRowF;

    float *dCsrValA,
      *dCsrValB,
      *dCsrValC,
      *dCsrValD,
      *dCsrValE;
      //*dCsrValF;

    int *dCsrRowPtrA,
      *dCsrRowPtrB,
      *dCsrRowPtrC,
      *dCsrRowPtrD,
      *dCsrRowPtrE;
      //*dCsrRowPtrF;

    int *dCsrColIndA,
      *dCsrColIndB,
      *dCsrColIndC,
      *dCsrColIndD,
      *dCsrColIndE;
      //*dCsrColIndF;

    int totalNnzA,
      totalNnzB,
      totalNnzC,
      baseD, totalNnzD,
      baseE, totalNnzE;
      //totalNnzF;

    float alpha = 1.0f;
    float beta = 1.0f;

    /*float* O;
    float* I;
    float* X;
    float* Y;
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    //float *D, *dD;
    float *E, *dE;
    float *F;//, *dF;*/
    float *dQ, *dX;
    float *dF;
    //float *Q;
    float *dError;

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    int *nnzTotalDevHostPtr = &totalNnzD;

    //uint16_t N = 10000;
    uint16_t N = 10000;
    uint16_t N2 = 2 * N;

    dim3 gridDim((N2 + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (N2 + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    dim3 gridDim1((N2 + BLOCK_DIM_X - 1) / BLOCK_DIM_X, 1);

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaStreamCreate(&stream_0));
    CHECK(cudaStreamCreate(&stream_1));

    CHECK(cudaEventRecord(start, 0));

    //double t;
    //Q = (float *) calloc(N2 * N2, sizeof(float));
    // Generate input
    /*X = (float *) calloc(N * N, sizeof(float));
    Y = (float *) calloc(N * N, sizeof(float));
    A = (float *) calloc(N2 * N2, sizeof(float));
    B = (float *) calloc(N2 * N2, sizeof(float));
    C = (float *) calloc(N2 * N2, sizeof(float));
    //D = (float *) calloc(N2 * N2, sizeof(float));
    E = (float *) calloc(N2 * N2, sizeof(float));
    F = (float *) calloc(N2 * N2, sizeof(float));*/

    CHECK(cudaMalloc((void **)&dX, sizeof(float) * N * N));

    curandGenerateUniform(prng, dX, N * N);

    // [I X]
    // [O I]
    CHECK(cudaMalloc((void **)&dQ, sizeof(float) * N2 * N2));
    CHECK(cudaMemset(dQ, 0, N2 * N2));
    gen_i<<<gridDim1, blockDim1>>>(dQ, N2);
    //CHECK(cudaMemcpy(Q, dQ, sizeof(float) * N2 * N2, cudaMemcpyDeviceToHost));
    //print_mat(Q, N2);

    CHECK_CUBLAS(cublasSetMatrix(N, N, sizeof(float), dX, N, (dQ + 2 * N * N), N2));
    /*for (uint16_t i = 0; i < N; i++) {
        printf("%u: %u\n", i, (N + i));
        CHECK(cudaMemcpyAsync((dQ + (N + i)), (dX + (i * N)), N * sizeof(float), cudaMemcpyDeviceToDevice, stream_0));
    }
    CHECK(cudaStreamSynchronize(stream_0));*/
    //CHECK(cudaMemcpy(Q, dQ, sizeof(float) * N2 * N2, cudaMemcpyDeviceToHost));
    //print_mat(Q, N2);

    CHECK(cudaFree(dX));
    //CHECK(cudaMalloc((void **)&dB, sizeof(float) * N2 * N2));
    //CHECK(cudaMemcpy(dA, dB, N2, cudaMemcpyDeviceToDevice));

    //gen_x(X, N);
    //copy_submat(Y, N, X, 0, 0, N);

    // XXX: Testing reduction
    /*
    float *dU, *dI;
    float *U = (float *)malloc(N * N * sizeof(float));
    float *I = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N*N; i++) {
        U[i] = 1;
        I[i] = 0;
    }
    //print_mat(U, N);
    //print_mat(I, N);

    CHECK(cudaMalloc((void **)&dU, N * N * sizeof(float)));
    CHECK(cudaMalloc((void **)&dI, N * N * sizeof(float)));

    CHECK(cudaMemcpy(dU, U, sizeof(float) * N * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dI, I, sizeof(float) * N * N, cudaMemcpyHostToDevice));

    float *dErf;
    CHECK(cudaMalloc((void **)&dErf, sizeof(float) * N * N));
    reduce0<<<256, 64, 256 * sizeof(float)>>>(dU, dI, dErf, N * N);
    float *erf = (float *) calloc(N * N, sizeof(float));
    CHECK(cudaMemcpy(erf, dErf, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0;
    for (int i = 0; i < N * N; i++) {
        sum += erf[i];
    }
    printf("%f\n", sum);*/
    //print_mat(erf, N);


/*
    t = seconds();
    gen_identity(A, N2);
    //printf("gen_identity: %f\n", seconds() - t);
    t = seconds();
    copy_submat(A, N2, X, 0, N, N);
    //printf("copy_submat: %f\n", seconds() - t);

    //t = seconds();
    gen_identity(C, N2);
    const_mult(Y, -1, 0, 0, N);
    copy_submat(C, N2, Y, 0, N, N);
    //printf("C: %f\n", seconds() - t);

    //t = seconds();
    gen_identity(B, N2);
    const_mult(B, -1, N, N, N2);
    const_mult(Y, -2, 0, 0, N);
    copy_submat(B, N2, Y, 0, N, N);
    //printf("B: %f\n", seconds() - t);

    free(X);
    free(Y);

    //t = seconds();
    gen_identity(F, N2);
    const_mult(F, -1, N, N, N2);
    //printf("F: %f\n", seconds() - t);
*/

    //float *F = (float *) calloc(N2 * N2, sizeof(float));
    //gen_identity(F, N2);
    //const_mult(F, -1, N, N, N2);
    CHECK(cudaDeviceSynchronize());
    //CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaEventRecord(stop, 0));

    CHECK(cudaEventSynchronize(stop));

    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for init:\t %3.1f ms\n", elapsed); 

    // A * B * C = D * C = E
    //mm(A, B, D, N2);
    //mm(D, C, E, N2);

    // cudaMemset(dA, 0, N2 * N2 * sizeof(float));

    CHECK(cudaEventRecord(start, 0));

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    // Construct a descriptor for the matrices
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    CHECK(cudaMalloc((void **)&dNnzPerRowA, sizeof(int) * N2));

    CHECK(cudaMalloc((void **)&dNnzPerRowC, sizeof(int) * N2));
    CHECK(cudaMalloc((void **)&dNnzPerRowE, sizeof(int) * N2));

    CHECK(cudaMalloc((void **)&dNnzPerRowB, sizeof(int) * N2));

    //CHECK(cudaMalloc((void **)&dA, sizeof(float) * N2 * N2));
    /*dense2csr(handle, A, dA, descr,
              dNnzPerRowA, dCsrValA, dCsrRowPtrA, dCsrColIndA,
              &totalNnzA, N2);

    dense2csr(handle, B, dA, descr,
              dNnzPerRowB, dCsrValB, dCsrRowPtrB, dCsrColIndB,
              &totalNnzB, N2);*/

    //CHECK(cudaMemcpy(dA, A, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));
    //free(A);

    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, N2, N2, descr, dQ,
                                N2, dNnzPerRowA, &totalNnzA));

    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalNnzA));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (N2 + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalNnzA));
    CHECK_CUSPARSE(cusparseSdense2csr(handle, N2, N2, descr, dQ, N2, dNnzPerRowA,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));


    // XXX: dQ: A -> C
    // [I -X]
    // [O  I]

    //gen_i<<<N2 * N2 / THREAD_COUNT, THREAD_COUNT>>>(dQ, N2);
    /*CHECK_CUBLAS(cublasSdgmm(handle_blas, CUBLAS_SIDE_LEFT, N, N,
                 (dQ + 2 * N * N), N2,
                 scalar, 0,
                 (dQ + 2 * N *N), N2));*/

    alpha = -1.0f;
    beta = 0.0f;
    CHECK_CUBLAS(cublasSgeam(handle_blas, trans_blas, trans_blas, N, N,
                 &alpha,
                 (dQ + 2 * N * N), N2,
                 &beta,
                 (dQ + 2 * N * N), N2,
                 (dQ + 2 * N *N), N2));

    //CHECK(cudaDeviceSynchronize());


    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, N2, N2, descr, dQ,
                                N2, dNnzPerRowC, &totalNnzC));
    CHECK(cudaMalloc((void **)&dCsrValC, sizeof(float) * totalNnzC));
    CHECK(cudaMalloc((void **)&dCsrRowPtrC, sizeof(int) * (N2 + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndC, sizeof(int) * totalNnzC));

    CHECK_CUSPARSE(cusparseSdense2csr(handle, N2, N2, descr, dQ, N2, dNnzPerRowC,
                                      dCsrValC, dCsrRowPtrC, dCsrColIndC));


    // XXX: dQ: A -> B
    // [I  X]
    // [O -I]

    alpha = -1.0f;
    CHECK_CUBLAS(cublasSgeam(handle_blas, trans_blas, trans_blas, N2, N,
                 &alpha,
                 (dQ + 2 * N * N), N2,
                 &beta,
                 (dQ + 2 * N * N), N2,
                 (dQ + 2 * N * N), N2));
    //CHECK(cudaDeviceSynchronize());

    // [I 2X]
    // [O -I]
    alpha = 2.0f;
    CHECK_CUBLAS(cublasSgeam(handle_blas, trans_blas, trans_blas, N, N,
                 &alpha,
                 (dQ + 2 * N * N), N2,
                 &beta,
                 (dQ + 2 * N * N), N2,
                 (dQ + 2 * N *N), N2));

    //CHECK(cudaMemcpy(Q, dQ, sizeof(float) * N2 * N2, cudaMemcpyDeviceToHost));
    //print_mat(Q, N2);

    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, N2, N2, descr, dQ,
                                N2, dNnzPerRowB, &totalNnzB));

    CHECK(cudaMalloc((void **)&dCsrValB, sizeof(float) * totalNnzB));
    CHECK(cudaMalloc((void **)&dCsrRowPtrB, sizeof(int) * (N2 + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndB, sizeof(int) * totalNnzB));
    CHECK_CUSPARSE(cusparseSdense2csr(handle, N2, N2, descr, dQ, N2, dNnzPerRowB,
                                      dCsrValB, dCsrRowPtrB, dCsrColIndB));
    CHECK(cudaFree(dQ));

    //unsigned dense = sizeof(float) * N2 * N2;
    //unsigned sparse = 
    //        sizeof(int) * N2 + sizeof(float) * totalNnzB +
    //        sizeof(int) * (N2 + 1) + sizeof(int) * totalNnzB;
    //printf("B (dense): %lu\n", dense);
    //printf("B (sparse): %lu\n", sparse);
    //printf("Savings: %f\n", (float) dense / (float) sparse);

    //CHECK(cudaFree(dA));
    CHECK(cudaMalloc((void **)&dCsrRowPtrD, sizeof(int) * (N2 + 1)));

    CHECK_CUSPARSE(cusparseXcsrgemmNnz(handle, trans, trans, N2, N2, N2, 
                        descr, totalNnzA, dCsrRowPtrA, dCsrColIndA,
                        descr, totalNnzB, dCsrRowPtrB, dCsrColIndB,
                        descr, dCsrRowPtrD, nnzTotalDevHostPtr));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));

    CHECK(cudaEventSynchronize(stop));

    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for init2:\t %3.1f ms\n", elapsed); 

    CHECK(cudaEventRecord(start, 0));
    if (NULL != nnzTotalDevHostPtr) {
        totalNnzD = *nnzTotalDevHostPtr;
    } else {
        cudaMemcpy(&totalNnzD, dCsrRowPtrD+N2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseD, dCsrRowPtrD, sizeof(int), cudaMemcpyDeviceToHost);
        totalNnzD -= baseD;
    }
    cudaMalloc((void**)&dCsrColIndD, sizeof(int)*totalNnzD);
    cudaMalloc((void**)&dCsrValD, sizeof(float)*totalNnzD);

    // Perform matrix-vector multiplication with the CSR-formatted matrices
    CHECK_CUSPARSE(cusparseScsrgemm(handle, trans, trans, N2, N2, N2,
                                    descr, totalNnzA, dCsrValA, dCsrRowPtrA, dCsrColIndA,
                                    descr, totalNnzB, dCsrValB, dCsrRowPtrB, dCsrColIndB,
                                    descr, dCsrValD, dCsrRowPtrD, dCsrColIndD));

    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));
    CHECK(cudaFree(dNnzPerRowA));
    CHECK(cudaFree(dCsrValB));
    CHECK(cudaFree(dCsrRowPtrB));
    CHECK(cudaFree(dCsrColIndB));
    CHECK(cudaFree(dNnzPerRowB));

    //CHECK(cudaMalloc((void **)&dC, sizeof(float) * N2 * N2));
    //CHECK(cudaMalloc((void **)&dE, sizeof(float) * N2 * N2));
    //CHECK(cudaMemcpy(dC, C, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(dE, E, sizeof(float) * N2 * N2, cudaMemcpyHostToDevice));


    //CHECK(cudaFree(dC));

    CHECK(cudaMalloc((void **)&dCsrRowPtrE, sizeof(int) * (N2 + 1)));

    CHECK_CUSPARSE(cusparseXcsrgemmNnz(handle, trans, trans, N2, N2, N2, 
                        descr, totalNnzD, dCsrRowPtrD, dCsrColIndD,
                        descr, totalNnzC, dCsrRowPtrC, dCsrColIndC,
                        descr, dCsrRowPtrE, &totalNnzE));

    if (NULL != nnzTotalDevHostPtr) {
        totalNnzE = *nnzTotalDevHostPtr;
    } else {
        cudaMemcpy(&totalNnzE, dCsrRowPtrE+N2, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseE, dCsrRowPtrE, sizeof(int), cudaMemcpyDeviceToHost);
        totalNnzE -= baseE;
    }
    cudaMalloc((void**)&dCsrColIndE, sizeof(int)*totalNnzE);
    cudaMalloc((void**)&dCsrValE, sizeof(float)*totalNnzE);

    CHECK_CUSPARSE(cusparseScsrgemm(handle, trans, trans, N2, N2, N2,
                                    descr, totalNnzD, dCsrValD, dCsrRowPtrD, dCsrColIndD,
                                    descr, totalNnzC, dCsrValC, dCsrRowPtrC, dCsrColIndC,
                                    descr, dCsrValE, dCsrRowPtrE, dCsrColIndE));

    CHECK(cudaMalloc((void **)&dQ, sizeof(float) * N2 * N2));
    CHECK_CUSPARSE(cusparseScsr2dense(handle, N2, N2, descr,
                                      dCsrValE, dCsrRowPtrE, dCsrColIndE,
                                      dQ, N2));

    //CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));

    CHECK(cudaEventSynchronize(stop));

    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for mm:\t %3.1f ms\n", elapsed); 

    // XXX: CHECK RESULTS
    
    CHECK(cudaEventRecord(start, 0));
    // Copy the result vector back to the host
    //CHECK(cudaMemcpy(Q, dQ, sizeof(float) * N2 * N2, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dCsrValC));
    CHECK(cudaFree(dCsrRowPtrC));
    CHECK(cudaFree(dCsrColIndC));
    CHECK(cudaFree(dNnzPerRowC));
    CHECK(cudaFree(dCsrValD));
    CHECK(cudaFree(dCsrRowPtrD));
    CHECK(cudaFree(dCsrColIndD));
    CHECK(cudaFree(dCsrValE));
    CHECK(cudaFree(dCsrRowPtrE));
    CHECK(cudaFree(dCsrColIndE));
    CHECK(cudaFree(dNnzPerRowE));
    
    CHECK(cudaMalloc((void **)&dF, sizeof(float) * N2 * N2));
    CHECK(cudaMemset(dF, 0, N2 * N2));
    gen_i<<<gridDim1, blockDim1>>>(dF, N2);
    alpha = -1.0f;
    CHECK_CUBLAS(cublasSgeam(handle_blas, trans_blas, trans_blas, N2, N,
                 &alpha,
                 (dF + 2 * N * N), N2,
                 &beta,
                 (dF + 2 * N * N), N2,
                 (dF + 2 * N * N), N2));

    CHECK(cudaMalloc((void **)&dError, sizeof(float) * N2 * N2));
    reduce6<64><<<256, 64, 256 * sizeof(float)>>>(dQ, dF, dError, N2);
    float *err = (float *) calloc(64, sizeof(float));
    CHECK(cudaMemcpy(err, dError, sizeof(float) * 64, cudaMemcpyDeviceToHost));
    float sum = 0;
    for (unsigned i = 0; i < 64; i++) {
        sum += err[i];
    }
    printf("Error: %.1f\n", sum);
    //print_mat(err, N2);
    free(err);
    //reduce6<64><<<gridDim, blockDim>>>(dQ, dF, dError, (unsigned int) N2);

    CHECK(cudaFree(dQ));
    CHECK(cudaFree(dF));
    CHECK(cudaFree(dError));

    //printf("Error: %d\n", mm_eq(Q, F, N2));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));

    CHECK(cudaEventSynchronize(stop));

    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken to verify:\t %3.1f ms\n", elapsed);

    CHECK(cudaEventRecord(start, 0));
    //free(C);
    //free(E);
    //free(F);
    //free(Q);

    //CHECK(cudaFree(dE));
    //CHECK(cudaFree(dF));

    //CHECK(cudaFree(dCsrValF));
    //CHECK(cudaFree(dCsrRowPtrF));
    //CHECK(cudaFree(dCsrColIndF));
    //CHECK(cudaFree(dNnzPerRowF));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));

    CHECK(cudaEventSynchronize(stop));

    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for freeing:\t %3.1f ms\n", elapsed); 

    return 0;
}

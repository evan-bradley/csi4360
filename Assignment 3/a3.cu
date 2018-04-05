#include "a3.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

int main(int argc, char **argv)
{
    cudaEvent_t verfStart, verfStop, start, stop;
    float total, elapsed, elapsed1;
    cudaStream_t stream_0, stream_1;

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, 100);

    cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cublasOperation_t trans_blas = CUBLAS_OP_N;
    cublasHandle_t handle_blas = 0;
    CHECK_CUBLAS(cublasCreate(&handle_blas));

    int *dNnzPerRowA,
      *dNnzPerRowB,
      *dNnzPerRowC,
      *dNnzPerRowE;

    float *dCsrValA,
      *dCsrValB,
      *dCsrValC,
      *dCsrValD,
      *dCsrValE;

    int *dCsrRowPtrA,
      *dCsrRowPtrB,
      *dCsrRowPtrC,
      *dCsrRowPtrD,
      *dCsrRowPtrE;

    int *dCsrColIndA,
      *dCsrColIndB,
      *dCsrColIndC,
      *dCsrColIndD,
      *dCsrColIndE;

    int totalNnzA,
      totalNnzB,
      totalNnzC,
      baseD, totalNnzD,
      baseE, totalNnzE;

    float alpha = 1.0f;
    float beta = 1.0f;

    float *dQ, *dX;
    float *dF;
    float *dError;

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    int *nnzTotalDevHostPtr = &totalNnzD;

    uint16_t N = 10000;
    uint16_t N2 = 2 * N;
    float sum = 0.0;

    dim3 blockDim1(1024, 1, 1); 
    dim3 gridDim1(ceil((float)(N2 * N2) / (float)blockDim1.x), 1, 1);

    CHECK(cudaEventCreate(&verfStart));
    CHECK(cudaEventCreate(&verfStop));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaStreamCreate(&stream_0));
    CHECK(cudaStreamCreate(&stream_1));

    CHECK(cudaEventRecord(verfStart, 0));
    CHECK(cudaEventRecord(start, 0));

    CHECK(cudaMalloc((void **)&dX, sizeof(float) * N * N));

    curandGenerateUniform(prng, dX, N * N);

    // [I X]
    // [O I]
    CHECK(cudaMalloc((void **)&dQ, sizeof(float) * N2 * N2));
    CHECK(cudaMemset(dQ, 0, N2 * N2));
    gen_i<<<gridDim1, blockDim1>>>(dQ, N2);

    CHECK_CUBLAS(cublasSetMatrix(N, N, sizeof(float), dX, N, (dQ + 2 * N * N), N2));

    CHECK(cudaFree(dX));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for init:\t %3.1f ms\n", elapsed); 

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

    alpha = -1.0f;
    beta = 0.0f;
    CHECK_CUBLAS(cublasSgeam(handle_blas, trans_blas, trans_blas, N, N,
                 &alpha,
                 (dQ + 2 * N * N), N2,
                 &beta,
                 (dQ + 2 * N * N), N2,
                 (dQ + 2 * N *N), N2));

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

    // [I 2X]
    // [O -I]
    alpha = 2.0f;
    CHECK_CUBLAS(cublasSgeam(handle_blas, trans_blas, trans_blas, N, N,
                 &alpha,
                 (dQ + 2 * N * N), N2,
                 &beta,
                 (dQ + 2 * N * N), N2,
                 (dQ + 2 * N *N), N2));

    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, N2, N2, descr, dQ,
                                N2, dNnzPerRowB, &totalNnzB));

    CHECK(cudaMalloc((void **)&dCsrValB, sizeof(float) * totalNnzB));
    CHECK(cudaMalloc((void **)&dCsrRowPtrB, sizeof(int) * (N2 + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndB, sizeof(int) * totalNnzB));
    CHECK_CUSPARSE(cusparseSdense2csr(handle, N2, N2, descr, dQ, N2, dNnzPerRowB,
                                      dCsrValB, dCsrRowPtrB, dCsrColIndB));

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

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for mm1:\t %3.1f ms\n", elapsed); 

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));
    CHECK(cudaFree(dNnzPerRowA));
    CHECK(cudaFree(dCsrValB));
    CHECK(cudaFree(dCsrRowPtrB));
    CHECK(cudaFree(dCsrColIndB));
    CHECK(cudaFree(dNnzPerRowB));

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

    CHECK_CUSPARSE(cusparseScsr2dense(handle, N2, N2, descr,
                                      dCsrValE, dCsrRowPtrE, dCsrColIndE,
                                      dQ, N2));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for mm2:\t %3.1f ms\n", elapsed); 

    // XXX: CHECK RESULTS
    
    CHECK(cudaEventRecord(start, 0));

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
    reduce<<<gridDim1, blockDim1, 2 * blockDim1.x * sizeof(float)>>>(dQ, dF, dError, N2);
    float *err = (float *) calloc(gridDim1.x, sizeof(float));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed1, start, stop));

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(err, dError, sizeof(float) * gridDim1.x, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken to memcpy:\t %3.1f ms\n", elapsed);

    for (unsigned i = 0; i < gridDim1.x; i++) {
        sum += err[i];
    }
    free(err);

    CHECK(cudaFree(dQ));
    CHECK(cudaFree(dF));
    CHECK(cudaFree(dError));

    printf("Time taken to verify:\t %3.1f ms\n", elapsed + elapsed1);

    CHECK(cudaEventRecord(start, 0));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for freeing:\t %3.1f ms\n", elapsed); 

    printf("Error: %.1f\n", sum);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(verfStop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&total, verfStart, verfStop));
    printf("Total execution time:\t %3.1f ms\n", total);

    return 0;
}

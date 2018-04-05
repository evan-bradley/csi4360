#include "a3.h"
#include <cublas_v2.h>
#include <curand_kernel.h>

void print_mat (float *A, uint16_t N) {
  for (uint16_t i = 0; i < N; i++) {
    for (uint16_t j = 0; j < N; j++) {
      printf("%.4f\t", A[i + N * j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
    cublasOperation_t trans = CUBLAS_OP_N;
    cudaEvent_t verfStart, verfStop, start, stop;
    float total, elapsed, elapsed1;

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, 100);

    float alpha = 1.0f;
    float beta = 0.0f;

    float *dA;
    float *dB;
    float *dC;
    float *dX;
    float *dF;
    float *dError;

    cublasHandle_t handle = 0;

    uint16_t N = 10000;
    uint16_t N2 = 2 * N;
    float sum = 0.0;

    dim3 blockDim1(1024, 1, 1); 
    dim3 gridDim1(ceil((float)(N2 * N2) / (float)blockDim1.x), 1, 1);

    //float *Q = (float *)malloc(N2 * N2 * sizeof(float));
    CHECK(cudaEventCreate(&verfStart));
    CHECK(cudaEventCreate(&verfStop));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(verfStart, 0));
    CHECK(cudaEventRecord(start, 0));
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
    CHECK(cudaMemset(dB, 0, N2 * N2));
    gen_i<<<gridDim1, blockDim1>>>(dB, N2);
    alpha = -1.0f;
    CHECK_CUBLAS(cublasSgeam(handle, trans, trans, N2, N,
                 &alpha,
                 (dB + 2 * N * N), N2,
                 &beta,
                 (dB + 2 * N * N), N2,
                 (dB + 2 * N * N), N2));
    CHECK_CUBLAS(cublasSetMatrix(N, N, sizeof(float), dX, N, (dB + 2 * N * N), N2));
    CHECK(cudaDeviceSynchronize());

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
    CHECK(cudaMemset(dC, 0, N2 * N2));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for init1:\t %3.1f ms\n", elapsed); 

    CHECK(cudaEventRecord(start, 0));
    alpha = 1.0f;
    beta = 1.0f;
    CHECK_CUBLAS(cublasSgemm_v2(handle, trans, trans, N2, N2, N2,
                             &alpha,
                             dA, N2,
                             dB, N2,
                             &beta,
                             dC, N2));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for mm1:\t %3.1f ms\n", elapsed); 

    CHECK(cudaEventRecord(start, 0));
    alpha = -1.0f;
    beta = 0.0f;
    CHECK_CUBLAS(cublasSgeam(handle, trans, trans, N, N,
                 &alpha,
                 (dA + 2 * N * N), N2,
                 &beta,
                 (dA + 2 * N * N), N2,
                 (dA + 2 * N *N), N2));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for init2:\t %3.1f ms\n", elapsed); 

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaFree(dB));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N2 * N2));
    CHECK(cudaMemset(dB, 0.0, N2 * N2));
    alpha = 1.0f;
    beta = 1.0f;
    CHECK_CUBLAS(cublasSgemm(handle, trans, trans, N2, N2, N2,
                             &alpha,
                             dC, N2,
                             dA, N2,
                             &beta,
                             dB, N2));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken for mm2:\t %3.1f ms\n", elapsed); 

    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dC));

    CHECK(cudaMalloc((void **)&dF, sizeof(float) * N2 * N2));
    CHECK(cudaMemset(dF, 0, N2 * N2));
    gen_i<<<gridDim1, blockDim1>>>(dF, N2);
    alpha = -1.0f;
    beta = 0.0f;
    CHECK_CUBLAS(cublasSgeam(handle, trans, trans, N2, N,
                 &alpha,
                 (dF + 2 * N * N), N2,
                 &beta,
                 (dF + 2 * N * N), N2,
                 (dF + 2 * N * N), N2));

    CHECK(cudaMalloc((void **)&dError, sizeof(float) * N2 * N2));
    reduce<<<gridDim1, blockDim1, 2 * blockDim1.x * sizeof(float)>>>(dB, dF, dError, N2);
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

    printf("Time taken to verify:\t %3.1f ms\n", elapsed + elapsed1);

    CHECK(cudaFree(dB));
    CHECK(cudaFree(dF));
    CHECK(cudaFree(dError));
    //free(Q);

    CHECK_CUBLAS(cublasDestroy(handle));

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

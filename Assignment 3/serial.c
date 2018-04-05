#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <memory.h>

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void zero (float *A, uint16_t N) {
  for (uint16_t i = 0; i < N; i++) {
    for (uint16_t j = 0; j < N; j++) {
      A[i * N + j] = 0;
    }
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

void matMul(int N, float *matrix1, float *matrix2, float *result)
{
    int sqr = N*N;
    memset(result, 0, sqr * sizeof(float));

    for (int ijk = 0; ijk < sqr; ijk++)
    {
        int i = ijk / N;
        int j = (ijk / N) % N;
        int k = ijk % N;
        result[i + N * j] += matrix1[i + N * k] * matrix2[k + N * j];
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
      error += fabsf(A[i + N * j] - B[i + N * j]);
    }
  }

  return error;
}

int main(int argc, char **argv) {
  double t;
  uint16_t N = 1000;
  uint16_t N2 = 2 * N;

  float* O;
  float* I;
  float* X;
  float* Y;

  float* A;
  float* B;
  float* C;
  float* D;
  float* E;
  float* F;

  t = seconds();
  //O = (float *) calloc(N * N, sizeof(float));
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

  gen_identity(F, N2);
  const_mult(F, -1, N, N, N2);
  printf("Init: %.1f\n", seconds() - t);

  t = seconds();
  mm(A, B, D, N2);
  //matMul(N2, A, B, D);
  printf("mm1: %.1f\n", seconds() - t);

  t = seconds();
  mm(D, C, E, N2);
  //matMul(N2, D, C, E);
  printf("mm2: %.1f\n",  seconds() - t);

  t = seconds();
  printf("Equal: %d\n", mm_eq(E, F, N2));
  printf("checking: %.1f\n", seconds() - t);

  //free(O);
  free(I);
  free(X);
  free(Y);
  free(A);
  free(B);
  free(C);
  free(D);
  free(E);
  free(F);
}

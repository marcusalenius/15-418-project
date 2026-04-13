// gemm.h

#pragma once

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "check.h"

// Given A [MxK], B [KxN], and C [MxN], computes C = A @ B [MxN].
// cuBLAS assumes column-major layout. So, it sees A^T [KxM], B^T [NxK],
// and C^T [NxM], all column-major. And computes C^T = B^T @ A^T [NxM],
// which is equal to C = A @ B [MxN].
inline void hgemm(
  cublasHandle_t handle,
  int M, int N, int K,
  const half* A, const half* B, half* C
) {
  const half alpha = __float2half(1.0f);
  const half beta  = __float2half(0.0f);

  CHECK_CUBLAS(
    // Signature: 
    //   (handle, 
    //    opA, opB, 
    //    m, n, k, 
    //    alpha, 
    //    A, lda, 
    //    B, ldb, 
    //    beta, 
    //    C, ldc)
    // Notes:
    //  - m = N, n = M, k = K -- the dimensions of the column-major C^T
    //  - opA = opB = CUBLAS_OP_N -- no transpose
    //  - lda = leading dimension A, etc for ldb and ldc
    cublasHgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, N,
      A, K,
      &beta,
      C, N
    )
  );     
}

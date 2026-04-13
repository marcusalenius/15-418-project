// check.h
// Error-checking macros for CUDA, cuBLAS, and NCCL.

#pragma once

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,       \
              status);                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_NCCL(call)                                                       \
  do {                                                                         \
    ncclResult_t res = (call);                                                 \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,         \
              ncclGetErrorString(res));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// This file measures the FLOPS that can be achieved by using gemm functions.
// This is intended to measure performance when the GPU is at different frequencies.

#define NUM_ITERATIONS 20
#define EASY_COPY

template <typename T>
void fill_random(T* arr, int dim_size) {
    for(int i = 0; i < dim_size * dim_size; i++) {
        arr[i] = (T)rand()/RAND_MAX;
    }
}

template <>
void fill_random(__half* arr, int dim_size) {
    for(int i = 0; i < dim_size * dim_size; i++) {
        arr[i] = __float2half(rand()/RAND_MAX);
    }
}

template <typename T>
void gemm(cublasHandle_t handle, int dim, T d_A, T d_B, T d_C) {
    printf("Unsupported Type\n");
}

template <>
void gemm(cublasHandle_t handle, int dim, __half* d_A, __half* d_B, __half* d_C) {
    printf("Half - %d\n", dim);
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(1.0f);
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
}

template <>
void gemm(cublasHandle_t handle, int dim, float* d_A, float* d_B, float* d_C) {
    printf("Float - %d\n", dim);
    float alpha = 1;
    float beta = 1;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
}

template <>
void gemm(cublasHandle_t handle, int dim, double* d_A, double* d_B, double* d_C) {
    printf("Double - %d\n", dim);
    double alpha = 1;
    double beta = 1;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
}

template <typename T>
void benchmark(int min_dim, int max_dim) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    T* h_A = (T*)malloc(max_dim * max_dim * sizeof(T));
    T* h_B = (T*)malloc(max_dim * max_dim * sizeof(T));
    T* h_C = (T*)malloc(max_dim * max_dim * sizeof(T));

    fill_random(h_A, max_dim);
    fill_random(h_B, max_dim);

    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, max_dim * max_dim * sizeof(T));
    cudaMalloc(&d_B, max_dim * max_dim * sizeof(T));
    cudaMalloc(&d_C, max_dim * max_dim * sizeof(T));

    cudaMemcpy(d_A, h_A, max_dim * max_dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, max_dim * max_dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, max_dim * max_dim * sizeof(T), cudaMemcpyHostToDevice);

    for (int dim = min_dim; dim <= max_dim; dim *= 2) {
        gemm<T>(handle, dim, d_A, d_B, d_C);
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    int min_dim = 128;
    int max_dim = 8192;
    benchmark<__half>(min_dim, max_dim);
    benchmark<float>(min_dim, max_dim);
    benchmark<double>(min_dim, max_dim);

    return 0;
}

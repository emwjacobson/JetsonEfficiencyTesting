#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// This file measures the FLOPS that can be achieved by using gemm functions.
// This is intended to measure performance when the GPU is at different frequencies.

#define RUNTIME 10 // In seconds

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
int gemm(cublasHandle_t handle, int dim, T d_A, T d_B, T d_C) {
    printf("Unsupported Type\n");
    return -1;
}

template <>
int gemm(cublasHandle_t handle, int dim, __half *d_A, __half *d_B, __half *d_C) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
        cudaDeviceSynchronize();
    }
    return i;
}

template <>
int gemm(cublasHandle_t handle, int dim, float *d_A, float *d_B, float *d_C) {
    float alpha = 1;
    float beta = 0;
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
        cudaDeviceSynchronize();
    }
    return i;
}

template <>
int gemm(cublasHandle_t handle, int dim, double *d_A, double *d_B, double *d_C) {
    double alpha = 1;
    double beta = 0;
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
        cudaDeviceSynchronize();
    }
    return i;
}



template <typename T>
void benchmark(int min_dim, int max_dim) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("Allocating array... ");
    T* h_A = (T*)malloc(max_dim * max_dim * sizeof(T));
    T* h_B = (T*)malloc(max_dim * max_dim * sizeof(T));
    T* h_C = (T*)malloc(max_dim * max_dim * sizeof(T));
    printf("Done\n");

    printf("Filling with random... ");
    fill_random(h_A, max_dim);
    fill_random(h_B, max_dim);
    printf("Done\n");

    printf("Allocating on GPU... ");
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, max_dim * max_dim * sizeof(T));
    cudaMalloc(&d_B, max_dim * max_dim * sizeof(T));
    cudaMalloc(&d_C, max_dim * max_dim * sizeof(T));
    printf("Done\n");

    printf("Copying to GPU... ");
    cudaMemcpy(d_A, h_A, max_dim * max_dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, max_dim * max_dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, max_dim * max_dim * sizeof(T), cudaMemcpyHostToDevice);
    printf("Done\n");


    printf("Running GEMM... \n");
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    float time_ms;
    double num_flop, final_time, final_flops;
    int num_iterations;
    for (int dim = min_dim; dim <= max_dim; dim *= 2) {
        printf("Matrix %d - ", dim);
        cudaEventRecord(gpu_start);

        num_iterations = gemm(handle, dim, d_A, d_B, d_C);

        cudaEventRecord(gpu_end);
        cudaEventSynchronize(gpu_end);
        cudaEventElapsedTime(&time_ms, gpu_start, gpu_end);

        // num_flop is the # of Floating Point Operations that should take place in a SINGLE matrix multiply
        num_flop = (unsigned long long)(dim * dim) * ((unsigned long long)(2 * dim) - 1);
        // final_time is the average time that it takes to do one matrix multiply
        final_time = ((time_ms / 1000.0) / num_iterations);
        // final_flops is number of Floating Point Operations Per Second that were achieved
        final_flops = (num_flop / (double) final_time);
        printf("%f FLOPS (%f seconds, %d iterations)\n", final_flops, (time_ms / 1000.0), num_iterations);
    }
    printf("Done\n");

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    int min_dim = 128;
    int max_dim = 8192;
    printf("Starting HALF\n");
    benchmark<__half>(min_dim, max_dim);
    printf("Done HALF\n\n");

    printf("Starting FLOAT\n");
    benchmark<float>(min_dim, max_dim);
    printf("Done FLOAT\n\n");

    printf("Starting DOUBLE\n");
    benchmark<double>(min_dim, max_dim);
    printf("Done DOUBLE\n\n");

    return 0;
}

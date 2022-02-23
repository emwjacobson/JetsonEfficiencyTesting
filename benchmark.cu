#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <thread>
#include <vector>
#include "jetson_clocks.hpp/jetson_clocks.hpp"

// This file measures the FLOPS that can be achieved by using gemm functions.
// This is intended to measure performance when the GPU is at different frequencies.

#define RUNTIME 20 // In seconds
#define USE_SOCKET

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


// START TENSOR FUNCTIONS

template <typename T>
int gemm(cublasHandle_t handle, int dim_rows, int dim_cols, T d_A, T d_B, T d_C) {
    printf("Unsupported Type\n");
    return -1;
}

template <>
int gemm(cublasHandle_t handle, int dim_rows, int dim_cols, __half *d_A, __half *d_B, __half *d_C) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_rows, dim_rows, dim_cols, &alpha, d_A, dim_rows, d_B, dim_rows, &beta, d_C, dim_rows);
        // cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
        cudaDeviceSynchronize();
    }
    return i;
}

template <>
int gemm(cublasHandle_t handle, int dim_rows, int dim_cols, float *d_A, float *d_B, float *d_C) {
    float alpha = 1;
    float beta = 0;
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_rows, dim_rows, dim_cols, &alpha, d_A, dim_rows, d_B, dim_rows, &beta, d_C, dim_rows);
        cudaDeviceSynchronize();
    }
    return i;
}

template <>
int gemm(cublasHandle_t handle, int dim_rows, int dim_cols, double *d_A, double *d_B, double *d_C) {
    double alpha = 1;
    double beta = 0;
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, dim, d_B, dim, &beta, d_C, dim);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_rows, dim_rows, dim_cols, &alpha, d_A, dim_rows, d_B, dim_rows, &beta, d_C, dim_rows);
        cudaDeviceSynchronize();
    }
    return i;
}

// END NON-TENSOR FUNCTIONS


// START TENSOR FUNCTIONS

template <typename T>
int gemm_tensor(cublasHandle_t handle, int dim_rows, int dim_cols, T d_A, T d_B, T d_C) {
    printf("Unsupported Type\n");
    return -1;
}

template <>
int gemm_tensor(cublasHandle_t handle, int dim_rows, int dim_cols, __half *d_A, __half *d_B, __half *d_C) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, CUDA_R_16F, dim, d_B, CUDA_R_16F, dim, &beta, d_C, CUDA_R_16F, dim, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_rows, dim_rows, dim_cols, &alpha, d_A, CUDA_R_16F, dim_rows, d_B, CUDA_R_16F, dim_rows, &beta, d_C, CUDA_R_16F, dim_rows, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
    }
    return i;
}

template <>
int gemm_tensor(cublasHandle_t handle, int dim_rows, int dim_cols, float *d_A, float *d_B, float *d_C) {
    float alpha = 1;
    float beta = 0;
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, CUDA_R_32F, dim, d_B, CUDA_R_32F, dim, &beta, d_C, CUDA_R_32F, dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_rows, dim_rows, dim_cols, &alpha, d_A, CUDA_R_32F, dim_rows, d_B, CUDA_R_32F, dim_rows, &beta, d_C, CUDA_R_32F, dim_rows, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
    }
    return i;
}

template <>
int gemm_tensor(cublasHandle_t handle, int dim_rows, int dim_cols, double *d_A, double *d_B, double *d_C) {
    double alpha = 1;
    double beta = 0;
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, CUDA_R_64F, dim, d_B, CUDA_R_64F, dim, &beta, d_C, CUDA_R_64F, dim, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_rows, dim_rows, dim_cols, &alpha, d_A, CUDA_R_64F, dim_rows, d_B, CUDA_R_64F, dim_rows, &beta, d_C, CUDA_R_64F, dim_rows, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
    }
    return i;
}

// END TENSOR FUNCTIONS


template <typename T>
std::string get_datatype(T* type) {
    return std::string("Unknown Datatype");
}

template <>
std::string get_datatype(__half* type) {
    return std::string("half");
}
template <>
std::string get_datatype(float* type) {
    return std::string("float");
}
template <>
std::string get_datatype(double* type) {
    return std::string("double");
}


template <typename T>
void benchmark(int sock, int min_rows, int max_rows, int step_rows, int min_cols, int max_cols, int step_cols, bool square) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int max_dim;
    if (square) {
        max_dim = max_rows;
    } else {
        max_dim = max(max_rows, max_cols);
    }
    int min_dim = min_rows;

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
    std::string msg;

    // Test tensor core and non-tensor core
    // https://forums.developer.nvidia.com/t/how-to-confirm-whether-tensor-core-is-working-or-not/70263/8

    // // Used when dealing with many different dimension matrices
    // for (int cur_rows = min_rows; cur_rows <= max_rows; cur_rows += step_rows) {
    //     for (int cur_cols = min_cols; cur_cols <= max_cols; cur_cols += step_cols) {
    //         // // This is the "non-tensor" version using the individual cublas<t>gemm functions
    //         // printf("Matrix %dx%d (Non-tensor) - ", cur_rows, cur_cols);
    //         // cudaEventRecord(gpu_start);

    //         // #ifdef USE_SOCKET
    //         // // START,datatype,dim_size,nontensor
    //         // // eg. START,half,256,nontensor
    //         // msg = "START," + get_datatype(h_A) + "," + std::to_string(cur_rows) + "," + std::to_string(cur_cols) + ",nontensor," + std::to_string(jetson_clocks::get_gpu_cur_freq());
    //         // send(sock, msg.c_str(), strlen(msg.c_str()), 0);
    //         // #endif

    //         // num_iterations = gemm(handle, cur_rows, cur_cols, d_A, d_B, d_C);

    //         // cudaEventRecord(gpu_end);
    //         // cudaEventSynchronize(gpu_end);
    //         // cudaEventElapsedTime(&time_ms, gpu_start, gpu_end);

    //         // // num_flop is the # of Floating Point Operations that should take place in a SINGLE matrix multiply
    //         // // num_flop = (unsigned long long)(dim * dim) * ((unsigned long long)(2 * dim) - 1);
    //         // num_flop = (unsigned long long)(cur_cols + cur_cols - 1) * (cur_rows * cur_rows);
    //         // // final_time is the average time that it takes to do one matrix multiply
    //         // final_time = ((time_ms / 1000.0) / num_iterations);
    //         // // final_flops is number of Floating Point Operations Per Second that were achieved
    //         // final_flops = (num_flop / (double) final_time);
    //         // printf("%f FLOPS (%f seconds, %d iterations)\n", final_flops, (time_ms / 1000.0), num_iterations);

    //         // #ifdef USE_SOCKET
    //         // msg = "DONE," + std::to_string(final_flops);
    //         // send(sock, msg.c_str(), strlen(msg.c_str()), 0);
    //         // #endif

    //         // printf("Small cooling between matrix size...\n");
    //         // jetson_clocks::set_fan_speed(255);
    //         // std::this_thread::sleep_for(std::chrono::milliseconds(15000));
    //         // jetson_clocks::set_fan_speed(0);
    //         // std::this_thread::sleep_for(std::chrono::milliseconds(4000));




    //         // This is the "tensor" version using the cublasGemmEx function
    //         printf("Matrix %dx%d (Tensor) - ", cur_rows, cur_cols);
    //         cudaEventRecord(gpu_start);

    //         #ifdef USE_SOCKET
    //         // START,datatype,dim_size,tensor
    //         // eg. START,half,256,tensor
    //         msg = "START," + get_datatype(h_A) + "," + std::to_string(cur_rows) + "," + std::to_string(cur_cols) + ",tensor," + std::to_string(jetson_clocks::get_gpu_cur_freq());
    //         send(sock, msg.c_str(), strlen(msg.c_str()), 0);
    //         #endif

    //         num_iterations = gemm_tensor(handle, cur_rows, cur_cols, d_A, d_B, d_C);

    //         cudaEventRecord(gpu_end);
    //         cudaEventSynchronize(gpu_end);
    //         cudaEventElapsedTime(&time_ms, gpu_start, gpu_end);

    //         // num_flop is the # of Floating Point Operations that should take place in a SINGLE matrix multiply
    //         // num_flop = (unsigned long long)(dim * dim) * ((unsigned long long)(2 * dim) - 1);
    //         num_flop = (unsigned long long)(cur_cols + cur_cols - 1) * (cur_rows * cur_rows);
    //         // final_time is the average time that it takes to do one matrix multiply
    //         final_time = ((time_ms / 1000.0) / num_iterations);
    //         // final_flops is number of Floating Point Operations Per Second that were achieved
    //         final_flops = (num_flop / (double) final_time);
    //         printf("%f FLOPS (%f seconds, %d iterations)\n", final_flops, (time_ms / 1000.0), num_iterations);

    //         #ifdef USE_SOCKET
    //         msg = "DONE," + std::to_string(final_flops);
    //         send(sock, msg.c_str(), strlen(msg.c_str()), 0);
    //         #endif

    //         printf("Small wait between matrix size...\n");
    //         // jetson_clocks::set_fan_speed(255);
    //         // std::this_thread::sleep_for(std::chrono::milliseconds(4000));
    //         // jetson_clocks::set_fan_speed(0);
    //         std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //     }
    // }


    // // Used when dealing with a square matrix
    for (int dim = min_dim; dim <= max_dim; dim += 64) {
    //     // // This is the "non-tensor" version using the individual cublas<t>gemm functions
    //     // printf("Matrix %d (Non-tensor) - ", dim);
    //     // cudaEventRecord(gpu_start);

    //     // #ifdef USE_SOCKET
    //     // // START,datatype,dim_size,nontensor
    //     // // eg. START,half,256,nontensor
    //     // msg = "START," + get_datatype(h_A) + "," + std::to_string(dim) + ",nontensor," + std::to_string(jetson_clocks::get_gpu_cur_freq());
    //     // send(sock, msg.c_str(), strlen(msg.c_str()), 0);
    //     // #endif

    //     // num_iterations = gemm(handle, dim, dim, d_A, d_B, d_C);

    //     // cudaEventRecord(gpu_end);
    //     // cudaEventSynchronize(gpu_end);
    //     // cudaEventElapsedTime(&time_ms, gpu_start, gpu_end);

    //     // // num_flop is the # of Floating Point Operations that should take place in a SINGLE matrix multiply
    //     // num_flop = (unsigned long long)(dim * dim) * ((unsigned long long)(2 * dim) - 1);
    //     // // final_time is the average time that it takes to do one matrix multiply
    //     // final_time = ((time_ms / 1000.0) / num_iterations);
    //     // // final_flops is number of Floating Point Operations Per Second that were achieved
    //     // final_flops = (num_flop / (double) final_time);
    //     // printf("%f FLOPS (%f seconds, %d iterations)\n", final_flops, (time_ms / 1000.0), num_iterations);

    //     // #ifdef USE_SOCKET
    //     // msg = "DONE," + std::to_string(final_flops);
    //     // send(sock, msg.c_str(), strlen(msg.c_str()), 0);
    //     // #endif

    //     // printf("Small cooling between matrix size...\n");
    //     // jetson_clocks::set_fan_speed(255);
    //     // std::this_thread::sleep_for(std::chrono::milliseconds(4000));
    //     // jetson_clocks::set_fan_speed(0);
    //     // std::this_thread::sleep_for(std::chrono::milliseconds(1000));





        // This is the "tensor" version using the cublasGemmEx function
        printf("Matrix %d (Tensor) - ", dim);
        cudaEventRecord(gpu_start);

        #ifdef USE_SOCKET
        // START,datatype,dim_size,tensor
        // eg. START,half,256,tensor
        msg = "START," + get_datatype(h_A) + "," + std::to_string(dim) + ",tensor," + std::to_string(jetson_clocks::get_gpu_cur_freq());
        send(sock, msg.c_str(), strlen(msg.c_str()), 0);
        #endif

        num_iterations = gemm_tensor(handle, dim, dim, d_A, d_B, d_C);

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

        #ifdef USE_SOCKET
        msg = "DONE," + std::to_string(final_flops);
        send(sock, msg.c_str(), strlen(msg.c_str()), 0);
        #endif

        printf("Small cooling between matrix size...\n");
        jetson_clocks::set_fan_speed(255);
        std::this_thread::sleep_for(std::chrono::milliseconds(4000));
        jetson_clocks::set_fan_speed(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
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

int connect_socket() {
    struct sockaddr_in serv_addr;
    char buffer[256];
    char * errorMsg;
    int sock;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Socket Creation Error\n");
        return -1;
    }
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8888);
    if (inet_pton(AF_INET, "192.168.8.112", &serv_addr.sin_addr) <= 0) {
        printf("Invalid address\n");
        return -1;
    }
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        errorMsg = strerror_r(errno, buffer, 256);
        printf("Connection Failed: %s\n", errorMsg);

        return -1;
    }
    return sock;
}

void benchmark_datatypes(int sock, int min_rows, int max_rows, int step_rows, int min_cols, int max_cols, int step_cols, bool square) {
    // printf("Starting HALF\n");
    // benchmark<__half>(sock, min_rows, max_rows, step_rows, min_cols, max_cols, step_cols, square);
    // printf("Done HALF\n\n");

    // printf("Cooling down Jetson between datatypes\n");
    // jetson_clocks::set_fan_speed(255);
    // std::this_thread::sleep_for(std::chrono::milliseconds(60000));
    // jetson_clocks::set_fan_speed(0);
    // std::this_thread::sleep_for(std::chrono::milliseconds(4000));


    printf("Starting FLOAT\n");
    benchmark<float>(sock, min_rows, max_rows, step_rows, min_cols, max_cols, step_cols, square);
    printf("Done FLOAT\n\n");

    // printf("Cooling down Jetson between datatypes\n");
    // jetson_clocks::set_fan_speed(255);
    // std::this_thread::sleep_for(std::chrono::milliseconds(60000));
    // jetson_clocks::set_fan_speed(0);
    // std::this_thread::sleep_for(std::chrono::milliseconds(4000));


    // printf("Starting DOUBLE\n");
    // benchmark<double>(sock, min_rows, max_rows, step_rows, min_cols, max_cols, step_cols, square);
    // printf("Done DOUBLE\n\n");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    int min_rows = 64;
    int max_rows = 2048;
    int step_rows = 64;

    int min_cols = 8;
    int max_cols = 2048;
    int step_cols = 8;
    // int min_cols = 0;
    // int max_cols = 0;
    // int step_cols = 0;

    #ifdef USE_SOCKET
    printf("Connecting to server... ");
    int sock = connect_socket();
    if (sock == -1) {
        printf("Error connecting to socket server.\n");
        return -1;
    }
    printf("Connected\n");
    #endif

    #ifdef USE_SOCKET
    benchmark_datatypes(sock, min_rows, max_rows, step_rows, min_cols, max_cols, step_cols, true);
    #else
    benchmark_datatypes(0, min_rows, max_rows, step_rows, min_cols, max_cols, step_cols, false);
    #endif

    return 0;
}

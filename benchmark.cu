#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/socket.h>
#include <arpa/inet.h>

// This file measures the FLOPS that can be achieved by using gemm functions.
// This is intended to measure performance when the GPU is at different frequencies.

#define RUNTIME 30 // In seconds
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

// END NON-TENSOR FUNCTIONS


// START TENSOR FUNCTIONS

template <typename T>
int gemm_tensor(cublasHandle_t handle, int dim, T d_A, T d_B, T d_C) {
    printf("Unsupported Type\n");
    return -1;
}

template <>
int gemm_tensor(cublasHandle_t handle, int dim, __half *d_A, __half *d_B, __half *d_C) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, CUDA_R_16F, dim, d_B, CUDA_R_16F, dim, &beta, d_C, CUDA_R_16F, dim, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
    }
    return i;
}

template <>
int gemm_tensor(cublasHandle_t handle, int dim, float *d_A, float *d_B, float *d_C) {
    float alpha = 1;
    float beta = 0;
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, CUDA_R_32F, dim, d_B, CUDA_R_32F, dim, &beta, d_C, CUDA_R_32F, dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
    }
    return i;
}

template <>
int gemm_tensor(cublasHandle_t handle, int dim, double *d_A, double *d_B, double *d_C) {
    double alpha = 1;
    double beta = 0;
    int i = 0;
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now() + std::chrono::seconds(RUNTIME);
    while (std::chrono::system_clock::now() < end) {
        i++;
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_A, CUDA_R_64F, dim, d_B, CUDA_R_64F, dim, &beta, d_C, CUDA_R_64F, dim, CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
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
void benchmark(int sock, int min_dim, int max_dim) {
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
    std::string msg;

    // Test tensor core and non-tensor core
    // https://forums.developer.nvidia.com/t/how-to-confirm-whether-tensor-core-is-working-or-not/70263/8

    // This is the "non-tensor" version using the individual cublas<t>gemm functions
    for (int dim = min_dim; dim <= max_dim; dim += 64) {
        printf("Matrix %d - ", dim);
        cudaEventRecord(gpu_start);

        #ifdef USE_SOCKET
        // START,datatype,dim_size,nontensor
        // eg. START,half,256,nontensor
        msg = "START," + get_datatype(h_A) + "," + std::to_string(dim) + ",nontensor";
        send(sock, msg.c_str(), strlen(msg.c_str()), 0);
        #endif

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

        #ifdef USE_SOCKET
        msg = "DONE," + std::to_string(final_flops);
        send(sock, msg.c_str(), strlen(msg.c_str()), 0);
        #endif
    }

    // This is the "tensor" version using cublasGemmEx
    for (int dim = min_dim; dim <= max_dim; dim += 64) {
        printf("Matrix %d - ", dim);
        cudaEventRecord(gpu_start);

        #ifdef USE_SOCKET
        // START,datatype,dim_size,tensor
        // eg. START,half,256,tensor
        msg = "START," + get_datatype(h_A) + "," + std::to_string(dim) + ",tensor";
        send(sock, msg.c_str(), strlen(msg.c_str()), 0);
        #endif

        num_iterations = gemm_tensor(handle, dim, d_A, d_B, d_C);

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
    if (inet_pton(AF_INET, "192.168.1.21", &serv_addr.sin_addr) <= 0) {
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

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    #ifdef USE_SOCKET
    printf("Connecting to server... ");
    int sock = connect_socket();
    if (sock == -1) {
        printf("Error connecting to socket server.\n");
        return -1;
    }
    printf("Connected\n");
    #endif

    int min_dim = 64;
    int max_dim = 4096;
    printf("Starting HALF\n");
    #ifdef USE_SOCKET
    benchmark<__half>(sock, min_dim, max_dim);
    #else
    benchmark<__half>(0, min_dim, max_dim);
    #endif
    printf("Done HALF\n\n");

    printf("Starting FLOAT\n");
    #ifdef USE_SOCKET
    benchmark<float>(sock, min_dim, max_dim);
    #else
    benchmark<float>(0, min_dim, max_dim);
    #endif
    printf("Done FLOAT\n\n");

    printf("Starting DOUBLE\n");
    #ifdef USE_SOCKET
    benchmark<double>(sock, min_dim, max_dim);
    #else
    benchmark<double>(0, min_dim, max_dim);
    #endif
    printf("Done DOUBLE\n\n");

    return 0;
}

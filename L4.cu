#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <chrono>

using namespace std;

#define BLOCK_SIZE 256
#define CUDA_CORES 768

__global__ void vectorAddShared(int* A, int* B, int* C, int n) {
    __shared__ int s_A[BLOCK_SIZE];
    __shared__ int s_B[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        s_A[threadIdx.x] = A[idx];
        s_B[threadIdx.x] = B[idx];
        __syncthreads();

        C[idx] = s_A[threadIdx.x] + s_B[threadIdx.x];
    }
}

void vectorAddCPU(int* A, int* B, int* C, int n) {
    for (int i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
}

int main() {
    int sizes[5] = {1000000, 5000000, 8000000, 10000000, 30000000};
    cout << "\nVector Addition Benchmark (Shared Memory)\n";
    cout << "--------------------------------------------------------------------------\n";
    cout << "| " << setw(12) << "Vector Size" 
         << " | " << setw(12) << "CPU Time(s)"
         << " | " << setw(12) << "GPU Time(s)"
         << " | " << setw(8) << "Speedup"
         << " | " << setw(10) << "Efficiency" << " |\n";
    cout << "--------------------------------------------------------------------------\n";

    for (int i = 0; i < 5; i++) {
        int N = sizes[i];
        int* h_A = (int*)malloc(N * sizeof(int));
        int* h_B = (int*)malloc(N * sizeof(int));
        int* h_C_CPU = (int*)malloc(N * sizeof(int));
        int* h_C_GPU = (int*)malloc(N * sizeof(int));

        for (int j = 0; j < N; ++j) {
            h_A[j] = rand() % 100;
            h_B[j] = rand() % 100;
        }

        // CPU time
        auto start_cpu = chrono::high_resolution_clock::now();
        vectorAddCPU(h_A, h_B, h_C_CPU, N);
        auto end_cpu = chrono::high_resolution_clock::now();
        chrono::duration<double> cpu_duration = end_cpu - start_cpu;
        double cpu_time = cpu_duration.count();

        // Allocate device memory
        int *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, N * sizeof(int));
        cudaMalloc((void**)&d_B, N * sizeof(int));
        cudaMalloc((void**)&d_C, N * sizeof(int));

        cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

        // GPU time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cudaEventRecord(start);
        vectorAddShared<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, start, stop);
        double gpu_time = gpu_time_ms / 1000.0;

        cudaMemcpy(h_C_GPU, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

        double speedup = cpu_time / gpu_time;
        double efficiency = speedup / CUDA_CORES;

        cout << "| " << setw(12) << N
             << " | " << setw(12) << fixed << setprecision(6) << cpu_time
             << " | " << setw(12) << fixed << setprecision(6) << gpu_time
             << " | " << setw(8) << fixed << setprecision(2) << speedup
             << " | " << setw(10) << fixed << setprecision(6) << efficiency
             << " |\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    cout << "--------------------------------------------------------------------------\n";
    return 0;
}

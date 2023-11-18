#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ float sumatoria;

__global__ void kernel(float *arr, float *accuaracy){
    atomicAdd(&accuaracy[0], arr[threadIdx.x]);
    __syncthreads();
    printf("%f \n", accuaracy[threadIdx.x]);
}

int main(){
    float *h_arr;
    float *d_arr;

    float *h_arr2;
    float *d_arr2;

    h_arr = (float *)malloc(16 * sizeof(float));
    h_arr2 = (float *)malloc(16 * sizeof(float));

    for (int i = 0; i < 16; i++)
    {
        h_arr[i] = 1;
        h_arr2[i] = 0;
    }
    
    cudaMalloc((void**)&d_arr, 16 * sizeof(float));
    cudaMalloc((void**)&d_arr2, 16 * sizeof(float));
    cudaMemcpy(d_arr2, h_arr2, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr, h_arr, 16 * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1,16>>>(d_arr, d_arr2);
    cudaDeviceSynchronize();
    cudaDeviceReset();
}
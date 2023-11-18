#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ float sumatoria;

__global__ void kernel(float *arr, float *accuaracy)
{
    atomicAdd(&accuaracy[0], 1/2);
    printf("%f \n", accuaracy[0]);
}

int main()
{

    int N = 112;
    float *h_arr;
    float *d_arr;

    float *h_arr2;
    float *d_arr2;

    h_arr = (float *)malloc(N * sizeof(float));
    h_arr2 = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        h_arr[i] = 1;
        h_arr2[i] = 0;
    }

    cudaMalloc((void **)&d_arr, N * sizeof(float));
    cudaMalloc((void **)&d_arr2, N * sizeof(float));
    cudaMemcpy(d_arr2, h_arr2, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, 112>>>(d_arr, d_arr2);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr2, d_arr2, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Final : %f", h_arr2[0]);

    cudaDeviceReset();
}
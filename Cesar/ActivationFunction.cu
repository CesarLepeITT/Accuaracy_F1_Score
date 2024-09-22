#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void ActivationFunction(float *semantica, float alpha, int nx, int ny)
{
    // Calculate global coordinates of the thread in the 2D block
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    printf("%i, %i, %i \n", ix, iy, tid);
    // Check if the index is within the data bounds
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        semantica[tid] = fdividef(1, 1 + exp(-1 * alpha * semantica[tid]));
        printf("%f, en %i \n", semantica[tid], tid);

    }
}

int main()
{
    // Size
    int nx, ny;

    ny = 1;
    nx = 1;

    // Memory size
    int sizeYPred;
    sizeYPred = sizeof(float) * nx;

    // Host initializations
    float alpha;
    alpha = 0.5F;

    // Device initialization
    float *y_pred;
    cudaMallocManaged((void **)&y_pred, sizeYPred);

    printf("Ingrese el valor de x: ");
    scanf("%f", &y_pred[0]);

    // Kernell call
    ActivationFunction<<<1, 1>>>(y_pred, alpha, nx, ny);
    cudaDeviceSynchronize();

    // Device reset
    cudaDeviceReset();
}

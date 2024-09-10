#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "AccuracyMultiClass.h"
#include "AccuracyMultiClass.cpp"
__global__ void ConfusionF1(float *yPred, float *yTrue, int nx, int ny, int *TP, int *FP, int *FN)
{
    // Calculate global coordinates of the thread in the 2D block
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    // Check if the index is within the data bounds
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        if (yPred[tid] == yTrue[ix])
        {
            atomicAdd(&TP[iy], 1);
        }
        else
        {
            atomicAdd(&FP[iy], 1);
            atomicAdd(&FN[iy], 1);
        }
    }
}

__global__ void F1ScoreMicro(float *f1Score, int ny, int *TP, int *FP, int *FN)
{
    /* Micro F1 Score */
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * ny + ix;

    if (tid < ny && ix < ny)
    {
        float y = 2 * TP[ix] + FP[ix] + FN[ix];

        y > 0 ? f1Score[ix] = fdividef(float(2 * TP[ix]), y) : f1Score[ix] = 0;
    }
}

int main()
{
    // Set up size
    int nx = 8;
    int ny = 1;
    int nm = nx * ny;

    // Size of memory
    int nBytesSemantica = sizeof(float) * nm;
    int nBytesTrue = sizeof(float) * nx;
    int nBytesF1Score = sizeof(float) * ny;
    int nBytesTPFPNP = sizeof(int) * ny;

    // Host mem allocation
    float *semantica;
    float *yTrue;
    float *f1Score;
    int *TP, *FP, *FN;

    cudaMallocManaged((void **)&semantica, nBytesSemantica);
    cudaMallocManaged((void **)&yTrue, nBytesTrue);
    cudaMallocManaged((void **)&f1Score, nBytesF1Score);
    cudaMallocManaged((void **)&TP, nBytesTPFPNP);
    cudaMallocManaged((void **)&FP, nBytesTPFPNP);
    cudaMallocManaged((void **)&FN, nBytesTPFPNP);

    // Host mem initialization
    FillingMatrices(semantica, nx, ny, 2);
    FillingVector(yTrue, nx, 2);
    FillingVector(f1Score, ny, 0);
    FillingVector(TP, ny, 0);
    FillingVector(FP, ny, 0);
    FillingVector(FN, ny, 0);

    printf("Semantica \n");
    imprimirMatriz(ny, nx, semantica);
    printf("Ytrue\n");
    imprimirVector(nx, yTrue);

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    ConfusionF1<<<grid, block>>>(semantica, yTrue, nx, ny, TP, FP, FN);
    cudaDeviceSynchronize();

    F1ScoreMicro<<<grid, block>>>(f1Score, ny, TP, FP, FN);
    cudaDeviceSynchronize();

    printf("TP\n");
    imprimirVector(ny, TP);
    printf("Fp\n");
    imprimirVector(ny, FP);
    printf("fn\n");
    imprimirVector(ny, FN);
    printf("f1\n");
    log10

    imprimirVector(ny, f1Score);
    cudaDeviceReset();
}

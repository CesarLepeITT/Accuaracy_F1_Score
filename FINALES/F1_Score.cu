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

__global__ void F1Score(float *f1Score, int ny, int *TP, int *FP, int *FN)
{
    // Implementar la división con
    // __global__ fdividef(float x, float y)
    // para evitar división sobre 0 y 
    // hacer pruebas unitarias para revisar que 
    // tabaje correctamente.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * ny + ix;

    if (tid < ny)
    {
         float precision;
        float recall;
        
        precision = TP[ix];
        precision = precision * 1 / (TP[ix] + FP[ix]);

        recall = TP[ix];
        recall = recall * 1 / (TP[ix] + FN[ix]);

        if(precision && recall){
            f1Score[ix] = 2 * precision * recall;
            f1Score[ix] = f1Score[ix] * 1 / (precision + recall);
        }
        else
            f1Score = 0;
        printf("%f \n", precision);
    }
}

int main()
{
    // Set up size
    int nx = 3;
    int ny = 3;
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
    FillingMatrices(semantica, nx, ny, 1);
    FillingVector(yTrue, nx, 1);
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

    F1Score<<<grid, block>>>(f1Score, ny, TP, FP, FN);
    cudaDeviceSynchronize();

    printf("TP\n");
    imprimirVector(ny, TP);
    printf("Fp\n");
    imprimirVector(ny, FP);
    printf("fn\n");
    imprimirVector(ny, FN);
    printf("f1\n");

    imprimirVector(ny, f1Score);
    cudaDeviceReset();
}

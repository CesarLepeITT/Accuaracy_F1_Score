#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;
__global__ void computeError(float *semantics, float *targetValues, float *fit, int nrow)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0;

    for (int i = 0; i < nrow; i++)
    {
        temp += (semantics[tid * nrow + i] - targetValues[i]) * (semantics[tid * nrow + i] - targetValues[i]);
    }
    temp = sqrt(temp / nrow);
    fit[tid] = temp;
}

// Hacer sigmoid directa y probar con nomas el exp
/*
__global__ void ActivationFunction(float *semantic, float *sigmoidSemantic, int nx, int ny)
{
    // sigmoidSemantic[tid] = fdividef(1, 1 + exp(-1 * alpha * semantic[tid]));
    // const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;
    if (tid < ny)
    {
        for (int i = 0; i < nx; i++)
        {
            unsigned int pos = tid * nx + i;
            if (pos < nx * ny)
            {
                sigmoidSemantic[pos] = 1.0 / (1 + exp(-1 * (semantic[pos])));
                //printf("tid: %i, pos: %i \n", tid, pos);
            }
        }
    }
}*/

__global__ void ActivationFunction(float *semantic, float * sigmoidSemantic, float alpha, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    // Check if the thread is within the matrix bounds
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        // Apply the sigmoid activation function to the current value
        sigmoidSemantic[tid] = fdividef(1, 1 + exp(-1 * alpha * semantic[tid]));

        printf("Tid:%i ix %i iy %i \n", tid, ix, iy);
    }
}


void rellenarMatriz(float *matriz, int filas, int columnas)
{
    for (int i = 0; i < filas; i++)
    {
        for (int j = 0; j < columnas; j++)
        {
            *(matriz + i * columnas + j) = static_cast<float>(rand()) / RAND_MAX; // Valor real entre 0 y 1
        }
    }
    matriz[0] = 0.0;
}

void imprimirMatriz(float *matriz, int filas, int columnas)
{
    for (int i = 0; i < filas; i++)
    {
        for (int j = 0; j < columnas; j++)
        {
            cout << *(matriz + i * columnas + j) << " ";
        }
        cout << endl;
    }
}

int main()
{
    // Size
    int nx, ny;

    ny = 3;
    nx = 4;

    // Memory size
    int sizeYPred;
    sizeYPred = sizeof(float) * nx;

    // Host initializations
    float alpha;
    alpha = 0.5F;

    // Device initialization
    float *semantica, *cpy;
    cudaMallocManaged((void **)&semantica, sizeYPred);
    cudaMallocManaged((void **)&cpy, sizeYPred);

    // Rellenar matrices
    rellenarMatriz(semantica, ny, nx);

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    ActivationFunction<<<grid, block>>>(semantica, cpy, 1, nx, ny);
    cudaDeviceSynchronize();

    // Ver datos
    printf("Semantica \n");
    imprimirMatriz(semantica, ny, nx);

    printf("Activacion \n");
    imprimirMatriz(cpy, ny, nx);

    // Device reset
    cudaDeviceReset();
}

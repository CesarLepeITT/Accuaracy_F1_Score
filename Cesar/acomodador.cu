#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void acomodar(int *y_true, int valor, int nx)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    if (tid < nx  && ix < nx)
    {
        if (y_true[ix] == valor)
            y_true[ix] = 1;
        else
            y_true[ix] = 0;
    }
}

void imprimitVector(int *vector, int nx)
{
    for (int i = 0; i < nx; i++)
    {
        if (i < nx - 1)
        {
            printf("%i ", vector[i]);
        }
        else
        {
            printf("%i \n", vector[i]);
        }
    }
}

void rellenar(int *vector, int valor1, int valor2, int nx)
{
    for (int i = 0; i < nx; i++)
    {
        if (i % 2 == 0)
        {
            vector[i] = valor1;
        }
        else
        {
            vector[1] = valor2;
        }
    }
}

int main()
{
    // Set up dimentions
    unsigned int nx;
    nx = 9;

    // Memory size
    unsigned int nBytesVect = sizeof(int) * nx;

    // Host memory initializartion
    int *y_true;
    int valor1, valor2;
    valor1 = 2;
    valor2 = 3;

    // Device memory initialization
    cudaMallocManaged((void **)&y_true, nBytesVect);
    rellenar(y_true, valor1, valor2, nx);

    // Visualizar y
    printf("Y true preprocesada\n");
    imprimitVector(y_true, nx);

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (1 + block.y - 1) / block.y);
    acomodar<<<grid, block>>>(y_true, valor1, nx);
    cudaDeviceSynchronize();

    // Check
    printf("Y true procesada, el valor seleccionado es %i \n", valor1);
    imprimitVector(y_true, nx);

    // Device reset
    cudaDeviceReset();
}
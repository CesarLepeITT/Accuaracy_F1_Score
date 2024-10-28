#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * @brief Converts an integer array into binary labels based on a reference value.
 *
 * This CUDA kernel processes a 1D array of integers (`outputArray`) and sets each 
 * element to 1 if it matches the specified `referenceValue`, or to 0 otherwise. 
 * It is designed to operate in parallel across multiple threads in a CUDA grid.
 *
 * @param outputArray Pointer to a 1D array of integers that will be modified to contain binary labels.
 * @param referenceValue The value against which each element in the array is compared.
 * @param arraySize The total number of elements in the array.
 *
 * @author César Lepe García
 * @date October 25, 2024
 */
__global__ void SetBinaryLabels(int *outputArray, int referenceValue, int arraySize)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * arraySize + ix;

    if (tid < arraySize  && ix < arraySize)
    {
        if (outputArray[ix] == referenceValue)
            outputArray[ix] = 1;
        else
            outputArray[ix] = 0;
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
    SetBinaryLabels<<<grid, block>>>(y_true, valor1, nx);
    cudaDeviceSynchronize();

    // Check
    printf("Y true procesada, el valor seleccionado es %i \n", valor1);

    int * copia;
    cudaMallocManaged((void**)&copia, nBytesVect);
    cudaMemcpy(copia, y_true, nBytesVect, cudaMemcpyDeviceToDevice);
    imprimitVector(copia, nx);

    imprimitVector(y_true, nx);

    // Device reset
    cudaDeviceReset();
}
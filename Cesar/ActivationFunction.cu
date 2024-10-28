#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

/**
 * @brief Aplica la función de activación sigmoide sobre una matriz bidimensional.
 *
 * Esta función de CUDA utiliza múltiples hilos para aplicar la función de activación sigmoide
 * a cada elemento de una matriz de tamaño `nx` x `ny`. La sigmoide se define como:
 * \f[
 *     \text{sigmoid}(x) = \frac{1}{1 + e^{-\alpha x}}
 * \f]
 *
 * @param semantic Puntero a un arreglo unidimensional que contiene los valores de la matriz.
 * @param alpha Parámetro de ajuste para la función sigmoide. Controla la pendiente de la curva sigmoide.
 * @param nx Número de columnas de la matriz.
 * @param ny Número de filas de la matriz.
 *
 * @author César Lepe García
 * @date 25 de octubre de 2024
 */
__global__ void ActivationFunction(float *semantic, float alpha, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    // Verifica si el hilo está dentro de los límites de la matriz
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        // Aplica la función de activación sigmoide sobre el valor actual
        semantic[tid] = fdividef(1, 1 + exp(-1 * alpha * semantic[tid]));
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
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    ActivationFunction<<<grid, block>>>(y_pred, alpha, nx, ny);
    cudaDeviceSynchronize();

    // Device reset
    cudaDeviceReset();
}

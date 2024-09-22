#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;
/**
 * @brief Aplica una función de activación sigmoide sobre una matriz de valores en paralelo utilizando la GPU.
 *
 * Esta función es una implementación de la función de activación sigmoide que se ejecuta en un kernel de CUDA.
 * Se aplica la operación a cada elemento de la matriz bidimensional `semantica`, cuyo tamaño es `nx` por `ny`.
 * La fórmula utilizada para cada elemento es:
 *
 *     semantica[tid] = 1 / (1 + exp(-alpha * semantica[tid]))
 *
 * Donde `alpha` es un parámetro de ajuste de la función sigmoide.
 *
 * @param semantica Un puntero a la memoria de la GPU que contiene los valores de entrada y donde se almacenarán
 *                  los resultados después de aplicar la función de activación sigmoide. Debe ser un arreglo
 *                  bidimensional aplanado de tamaño `nx * ny`.
 * @param alpha     Un valor flotante que controla la "pendiente" de la función sigmoide.
 * @param nx        El número de columnas en la matriz `semantica`.
 * @param ny        El número de filas en la matriz `semantica`.
 *
 * @details
 * La función utiliza el paralelismo de CUDA para ejecutar múltiples hilos, donde cada hilo calcula un índice
 * basado en su identificación de bloque (`blockIdx`) e hilo (`threadIdx`). Cada hilo procesa un elemento en
 * la matriz `semantica`, siempre y cuando el índice calculado esté dentro de los límites de la matriz.
 *
 * El operador `fdividef` se usa para realizar una división en punto flotante de manera eficiente en la GPU.
 *
 * @note Es importante asegurarse de que el número total de hilos lanzados sea suficiente para cubrir el
 *       tamaño completo de la matriz `semantica` (nx * ny).
 *
 * @note La función sigmoide aplicada es una forma comúnmente utilizada en redes neuronales para normalizar
 *       las salidas en un rango de (0, 1).
 */
__global__ void ActivationFunction(float *semantica, float alpha, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    // Verifica si el hilo está dentro de los límites de la matriz
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        // Aplica la función de activación sigmoide sobre el valor actual
        semantica[tid] = fdividef(1, 1 + exp(-1 * alpha * semantica[tid]));
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

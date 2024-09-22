#include "./Cesar.h"

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

/*!
* \fn       __global__ void accuracyScore(float *y_pred, float *y_true,  float *accuracies, int nx, int ny);
* \brief    Compute the accuracy score
* \param    float *y_pred: array of predicted values
* \param    float *y_true: array of the target values
* \param    float *accuracies: array that stores the calculated accuracies
* \param    int nx: number of columns in y_pred and y_true
* \param    int ny: number of rows in y_pred
* \date     21/sept/2024
* \author   César Lepe Garcia
* \file     GsgpCuda.cpp
*/

__global__ void accuracyScore(float *y_pred, float *y_true,  float *accuracies, int nx, int ny)
{
    // Calculate global coordinates of the thread in the 2D block
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    // Check if the index is within the data bounds
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        // Compare predicted and true values (TP, TN)
        if (y_pred[tid] == y_true[ix])
        {
            // Calculate the contribution to accuracy and add it to the accuracies array
            float sum = 1 / (float)nx;
            atomicAdd(&accuracies[iy], sum);
        }
    }
}

/**
 * @brief Calcula el error cuadrático entre dos matrices de valores (predicciones y valores reales) en paralelo utilizando la GPU.
 *
 * Esta función implementa la parte del cálculo del "Error Cuadrático Medio" (Root Mean Squared Error - RMSE) en un kernel de CUDA.
 * El RMSE mide la diferencia promedio entre los valores predichos (`semantica`) y los valores reales (`yTrue`).
 * En este kernel, se calcula el cuadrado de la diferencia entre cada elemento de `semantica` y `yTrue` para cada posición correspondiente.
 *
 * La fórmula aplicada por cada hilo es:
 *
 *     semantica[tid] = (semantica[tid] - yTrue[tid])^2
 *
 * @param semantica Un puntero a la memoria de la GPU que contiene los valores predichos. Los resultados de los errores cuadrados
 *                  se almacenarán en el mismo arreglo.
 * @param yTrue     Un puntero a la memoria de la GPU que contiene los valores reales.
 * @param nx        El número de columnas en la matriz `semantica` y `yTrue`.
 * @param ny        El número de filas en la matriz `semantica` y `yTrue`.
 *
 * @details
 * La función paraleliza el cálculo utilizando hilos de CUDA, donde cada hilo calcula la diferencia cuadrática de un solo elemento
 * entre las matrices `semantica` y `yTrue`. El índice del hilo se determina a partir de las identificaciones de bloque (`blockIdx`)
 * e hilo (`threadIdx`).
 *
 * Es importante que el número de hilos lanzados cubra todo el rango de la matriz (`nx * ny`).
 *
 * @note El cálculo completo del RMSE requiere que la suma de los valores obtenidos en este kernel sea dividida por el número de elementos
 *       y luego se extraiga la raíz cuadrada, lo cual no se realiza en este kernel. Esta función solo computa el cuadrado de las diferencias.
 */
__global__ void RMSE(float *semantica, float *yTrue, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;
    // Verifica si el hilo está dentro de los límites de la matriz
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        // Calcula el error cuadrático para el elemento actual

        semantica[tid] = powf(semantica[tid] - yTrue[ix], 2);
    }
}
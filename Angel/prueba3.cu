#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

// cada hilo evalua un individuo y por ende cada hilo ocupa acceder a los datos reales (true values)
// por ello, almacenaremos por bloque una porcion de los datos del array con los true values en la compartida, cada hilo (dentro del bloque)
// podra acceder a ella para hacer sus calculos, comparando la data correspondiente del individuo que le fue asignado
__global__ void accuracy_score(float *_true, float *y_pred, bool truE, int m, float *accuaracy)
{
    // Operaciones de comparacion
    if (y_pred[m * threadIdx.y + threadIdx.x] == _true[threadIdx.x])
    {
        atomicAdd(&accuaracy[threadIdx.y], 1);
        __syncthreads();
        printf("x: %i, y:%i score: %f \n", threadIdx.x, threadIdx.y, accuaracy[threadIdx.y]);
    }
    if (threadIdx.x == m - 1)
        accuaracy[threadIdx.y] /= m;
}

void FillingMatrices(float *matrix, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int e = 0; e < m; e++)
            matrix[i * m + e] = 1;
}

void Predictions(float *vector, int m, float num)
{
    for (int i = 0; i < m; i++)
    {
        vector[i] = num;
    }
}

int main()
{
    // predicciones y valores esperados
    float *predictions;
    float *targValues;
    float *accuaracy;
    float *dpredictions;
    float *dtargValues;
    float *daccuaracy;

    // Matriz con 5 individuos y 6 columnas de datos (el array de valores esperados es de 6 elementos)
    int n = 16;
    int m = 16;
    int nm = n * m;

    // Sizes
    int sizeMatrix = nm * sizeof(float);
    int sizeVects = n * sizeof(float);

    //
    cudaMallocHost((void **)&accuaracy, sizeVects);
    cudaMallocHost((void **)&predictions, sizeMatrix);
    cudaMallocHost((void **)&targValues, sizeVects);

    cudaMalloc((void **)&daccuaracy, sizeVects);
    cudaMalloc((void **)&dtargValues, sizeVects);
    cudaMalloc((void **)&dpredictions, sizeMatrix);

    // Inicializar matrices
    FillingMatrices(predictions, n, m);
    Predictions(targValues, m, 1);
    Predictions(accuaracy, n, 0);

    // memcpy
    cudaMemcpy(daccuaracy, accuaracy, sizeVects, cudaMemcpyHostToDevice);
    cudaMemcpy(dtargValues, targValues, sizeVects, cudaMemcpyHostToDevice);
    cudaMemcpy(dpredictions, predictions, sizeMatrix, cudaMemcpyHostToDevice);

    dim3 block(m, n);
    accuracy_score<<<1, block>>>(targValues, predictions, false, m, accuaracy);
    cudaDeviceSynchronize();

    cudaMemcpy(accuaracy, daccuaracy, sizeVects, cudaMemcpyDeviceToHost);

    printf("[");
    for (int i = 0; i < n; i++)
    {
        if (i != n - 1)
           printf("%f, ", accuaracy[i]);
        else
            printf("%f", accuaracy[i]);
    }
    printf("]\n");

    cudaFree(accuaracy);
    cudaFree(targValues);
    cudaFree(predictions);
    cudaDeviceReset();
}
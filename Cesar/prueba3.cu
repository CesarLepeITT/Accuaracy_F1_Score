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
    int n = 1024;
    int m = 1024;
    int nm = n * m;

    // Sizes
    int sizeMatrix = nm * sizeof(float);
    int sizeVects = n * sizeof(float);

    // host
    accuaracy = (float *)malloc(sizeVects);
    targValues = (float *)malloc(sizeVects);
    predictions = (float *)malloc(sizeMatrix);
    //cudaMallocHost((void **)&predictions, sizeMatrix);
    //cudaMallocHost((void **)&targValues, sizeVects);

    // device
    cudaMalloc((void **)&daccuaracy, sizeVects);
    cudaMalloc((void **)&dtargValues, sizeVects);
    cudaMalloc((void **)&dpredictions, sizeMatrix);

    // Inicializar matrices host
    FillingMatrices(predictions, n, m);
    Predictions(targValues, m, 1);
    Predictions(accuaracy, n, 9999);

    // memcpy htd
    cudaMemcpy(daccuaracy, accuaracy, sizeVects, cudaMemcpyHostToDevice);
    cudaMemcpy(dtargValues, targValues, sizeVects, cudaMemcpyHostToDevice);
    cudaMemcpy(dpredictions, predictions, sizeMatrix, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Kernell call
    int dimx = 64;
    int dimy = 64;

    dim3 block(dimx, dimy);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    accuracy_score<<<block, grid>>>(targValues, predictions, false, m, daccuaracy);
    cudaDeviceSynchronize();

    cudaMemcpy(daccuaracy, accuaracy, sizeVects, cudaMemcpyDeviceToHost);
    cudaMemcpy(dtargValues, targValues, sizeVects, cudaMemcpyDeviceToHost);
    cudaMemcpy(dpredictions, predictions, sizeMatrix, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("[");
    for (int i = 0; i < n; i++)
    {
        if (i != n - 1)
            printf("%f, ", accuaracy[i]);
        else
            printf("%f", accuaracy[i]);
    }
    printf("]\n");

    cudaDeviceReset();
}
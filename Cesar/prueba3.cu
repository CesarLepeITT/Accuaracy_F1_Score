#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

// cada hilo evalua un individuo y por ende cada hilo ocupa acceder a los datos reales (true values)
// por ello, almacenaremos por bloque una porcion de los datos del array con los true values en la compartida, cada hilo (dentro del bloque)
// podra acceder a ella para hacer sus calculos, comparando la data correspondiente del individuo que le fue asignado
__device__ float sumatoria;
__global__ void accuracy_score(float *_true, float *y_pred, bool truE, int m, float *accuaracy)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Operaciones de comparacion
    if (y_pred[m * threadIdx.y + threadIdx.x] == _true[threadIdx.x])
    {   
        float sum = 1/m;
        atomicAdd(&accuaracy[threadIdx.y], sum);
        //printf("x: %i, y:%i score: %f ", threadIdx.x, threadIdx.y, accuaracy[threadIdx.y]);
        //printf(" %f \n",accuaracy[threadIdx.y]);
    }
    /*if (threadIdx.x == m - 1){
        accuaracy[threadIdx.y] /= m;
       // printf("a  %f",accuaracy[threadIdx.y]);
    }*/
    __syncthreads();
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
    int ny = 128;
    int nx = 128;
    int nm = ny * nx;

    // Sizes
    int sizePredictions = nm * sizeof(float);
    int sizeTargetValues = nx * sizeof(float);
    int sizeAccuracy = ny * sizeof(float);

    // host
    accuaracy = (float *)malloc(sizeAccuracy);
    targValues = (float *)malloc(sizeTargetValues);
    predictions = (float *)malloc(sizePredictions);

    // device
    cudaMalloc((void **)&daccuaracy, sizeAccuracy);
    cudaMalloc((void **)&dtargValues, sizeTargetValues);
    cudaMalloc((void **)&dpredictions, sizePredictions);

    // Inicializar matrices host
    FillingMatrices(predictions, ny, nx);
    Predictions(targValues, nx, 1);
    Predictions(accuaracy, ny, 0);
    cudaDeviceSynchronize();

    // memcpy htd
    cudaMemcpy(daccuaracy, accuaracy, sizeAccuracy, cudaMemcpyHostToDevice);
    cudaMemcpy(dtargValues, targValues, sizeTargetValues, cudaMemcpyHostToDevice);
    cudaMemcpy(dpredictions, predictions, sizePredictions, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    accuracy_score<<<grid, block>>>(dtargValues, dpredictions, false, nx, daccuaracy);
    cudaDeviceSynchronize();

    cudaMemcpy(accuaracy, daccuaracy, sizeAccuracy, cudaMemcpyDeviceToHost);
    cudaMemcpy(targValues, dtargValues, sizeTargetValues, cudaMemcpyDeviceToHost);
    cudaMemcpy(predictions, dpredictions, sizePredictions, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("[");
    for (int i = 0; i < ny; i++)
    {
        if (i != ny - 1)
            printf("%f, ", accuaracy[i]);
        else
            printf("%f", accuaracy[i]);
    }
    printf("]\n");

    cudaDeviceReset();
}
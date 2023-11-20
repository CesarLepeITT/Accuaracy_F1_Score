#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

// cada hilo evalua un individuo y por ende cada hilo ocupa acceder a los datos reales (true values)
// por ello, almacenaremos por bloque una porcion de los datos del array con los true values en la compartida, cada hilo (dentro del bloque)
// podra acceder a ella para hacer sus calculos, comparando la data correspondiente del individuo que le fue asignado
__device__ float sumatoria;
__global__ void accuracy_score(float *_true, float *y_pred, bool truE, int nx, int ny, float *accuaracy)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // 0 a 15
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y; // o 1 15
    unsigned int idx = iy * nx + ix;                         // Matriz
    unsigned int posx = idx - (iy * nx);

    // printf("ix: %i, iy:%i score: %i, idx%i \n", ix, iy, posx, idx);
    //  Operaciones de comparacion
    if (y_pred[idx] == _true[posx] && idx < nx * ny) // Corregir nx * nx a nx * ny
    {
        float sum = 1 / nx;
        atomicAdd(&accuaracy[iy], 1 / (float)nx); 
        //atomicAdd(reinterpret_cast<int*>(&accuaracy[iy]), __float_as_int(1 / nx)); 
        //accuaracy[iy] /= nx;      
        // printf("ix: %i, iy:%i posx: %i, idx%i, score: %f \n", ix, iy, posx, idx, accuaracy[iy]);
    }
   // if (posx == nx - 1)
   // {
    //    accuaracy[iy] /= nx;
   //     //printf("real score %f\n",accuaracy[iy]);
    //}
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
        if (i % 2 == 0)
            vector[i] = num;
        else 
            vector[i] = 5;
    }
}
void VectorVacio(float *vector, int m, float num)
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
    int ny = 5;
    int nx = 5;
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
    VectorVacio(accuaracy, ny, 0);
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
    accuracy_score<<<grid, block>>>(dtargValues, dpredictions, false, nx, ny, daccuaracy);
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
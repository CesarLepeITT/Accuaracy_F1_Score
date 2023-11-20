#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void accuracy_score(float *_true, float *y_pred, int nx, int ny, float *accuaracy)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; 
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y; 
    unsigned int idx = iy * nx + ix;                         
    unsigned int posx = idx - (iy * nx);

    if (y_pred[idx] == _true[posx] && idx < nx * ny) 
    {
        float sum = 1 / nx;
        atomicAdd(&accuaracy[iy], 1 / (float)nx);
    }
    __syncthreads();
}

void FillingMatrices(float *matrix, int n, int m)
{
    // for (int i = 0; i < n; i++)
    //     for (int e = 0; e < m; e++)
    //         matrix[i * m + e] = 1;
    matrix[0] = 1;
    matrix[0] = 1;
    matrix[1] = 0;
    matrix[2] = 1;
    matrix[3] = 1;
    matrix[4] = 0;
    matrix[5] = 0;
    matrix[6] = 1;
    matrix[7] = 1;
    matrix[8] = 1;
    matrix[9] = 0;
}

void Predictions(float *vector, int m, float num)
{
    //for (int i = 0; i < m; i++)
    //{
    //    if (i % 2 == 0)
     //       vector[i] = num;
     //   else
    //        vector[i] = 5;
    //}

    vector[0] = 1;
    vector[1] = 0;
    vector[2] = 1;
    vector[3] = 1;
    vector[4] = 0;
    vector[5] = 1;
    vector[6] = 0;
    vector[7] = 1;
    vector[8] = 1;
    vector[9] = 0;
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
    float *predictions;
    float *targValues;
    float *accuaracy;
    float *dpredictions;
    float *dtargValues;
    float *daccuaracy;

    int ny = 1;
    int nx = 10;
    int nm = ny * nx;

    int sizePredictions = nm * sizeof(float);
    int sizeTargetValues = nx * sizeof(float);
    int sizeAccuracy = ny * sizeof(float);

    accuaracy = (float *)malloc(sizeAccuracy);
    targValues = (float *)malloc(sizeTargetValues);
    predictions = (float *)malloc(sizePredictions);

    cudaMalloc((void **)&daccuaracy, sizeAccuracy);
    cudaMalloc((void **)&dtargValues, sizeTargetValues);
    cudaMalloc((void **)&dpredictions, sizePredictions);

    FillingMatrices(predictions, ny, nx);
    Predictions(targValues, nx, 1);
    VectorVacio(accuaracy, 1, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(daccuaracy, accuaracy, sizeAccuracy, cudaMemcpyHostToDevice);
    cudaMemcpy(dtargValues, targValues, sizeTargetValues, cudaMemcpyHostToDevice);
    cudaMemcpy(dpredictions, predictions, sizePredictions, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    accuracy_score<<<grid, block>>>(dtargValues, dpredictions, nx, ny, daccuaracy);
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
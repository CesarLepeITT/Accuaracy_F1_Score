#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
//TODO: Documentar el codigo
//TODO: Probar un problema real con el codigo
//TODO: Probar beneficios de usar pinned memory
//TODO: Probar beneficios de usar managed memory
//TODO: Optimizar rendimiento general

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
void PrintVect(float *vect, int ny){
        printf("[");
    for (int i = 0; i < ny; i++)
    {
        if (i != ny - 1)
            printf("%f, ", vect[i]);
        else
            printf("%f", vect[i]);
    }
    printf("]\n");
}
int main()
{
    // Set up dimensions
    int ny = 2048; 
    int nx = 2048; 
    int nm = ny * nx; 

    float *predictions, *targValues, *accuaracy;
    float *dpredictions, *dtargValues, *daccuaracy;

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

    cudaMemcpy(daccuaracy, accuaracy, sizeAccuracy, cudaMemcpyHostToDevice);
    cudaMemcpy(dtargValues, targValues, sizeTargetValues, cudaMemcpyHostToDevice);
    cudaMemcpy(dpredictions, predictions, sizePredictions, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    accuracy_score<<<grid, block>>>(dtargValues, dpredictions, nx, ny, daccuaracy);
    cudaDeviceSynchronize();

    cudaMemcpy(accuaracy, daccuaracy, sizeAccuracy, cudaMemcpyDeviceToHost);

    //PrintVect(accuaracy, ny);

    cudaDeviceReset();
}
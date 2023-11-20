#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
//TODO: Documentar el codigo
//TODO: Probar un problema real con el codigo
//TODO: Probar beneficios de usar pinned memory
//TODO: Probar beneficios de usar managed memory
//TODO: Optimizar rendimiento general

__global__ void f1_score(float *y_true, float *y_pred, float *f1_score, int nx, int ny, int *aux)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;
    unsigned int cesar = iy * 2;
    unsigned int segunda = cesar + 1;
    unsigned int tercera = cesar + 2;
    printf("ix%i iy%i tid %i p%i, s%i , t%i\n", ix, iy, tid, cesar, segunda, tercera);
  /* if (tid < nx * ny)
    {
        if (y_pred[tid] == 1 && y_true[ix] == 1) // TP
        {
            atomicAdd(&aux[cesar], 1);
        }
        if (y_pred[tid] == 1 && y_true[ix] == 0) // FP
        {
            atomicAdd(&aux[segunda], 1);
        }
        if (y_pred[tid] == 0 && y_true[ix] == 1) // FN
        {
            atomicAdd(&aux[tercera], 1);
        }
        if (tid == nx - 1)
        {
            unsigned int a = aux[cesar];
            unsigned int b = aux[segunda];
            unsigned int c = aux[tercera];
            unsigned int x = (a + b) * (a + c);
            if(x == 0) 
                x = 1;
            float r = 2 * a / x;
            f1_score[iy] = r;
        }
    }*/
}

void FillingMatrices(float *matrix, float value ,int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int e = 0; e < m; e++)
            matrix[i * m + e] = value;
}
void FillingMatrices(int *matrix, int value ,int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int e = 0; e < m; e++)
            matrix[i * m + e] = value;
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
    int ny = 2; 
    int nx = 2; 
    int nm = ny * nx; 

    float *h_predictions, *h_targetValues, *h_accuracy;
    int *h_aux;
    float *d_predictions, *d_targetValues, *d_accuracy;
    int *d_aux;

    int sizePredictions = nm * sizeof(float);
    int sizeTargetValues = nx * sizeof(float);
    int sizeAccuracy = ny * sizeof(float);
    int sizeAux = 3 * ny * sizeof(int);

    h_accuracy = (float *)malloc(sizeAccuracy);
    h_targetValues = (float *)malloc(sizeTargetValues);
    h_predictions = (float *)malloc(sizePredictions);
    h_aux = (int*)malloc(sizeAux);

    cudaMalloc((void **)&d_accuracy, sizeAccuracy);
    cudaMalloc((void **)&d_targetValues, sizeTargetValues);
    cudaMalloc((void **)&d_predictions, sizePredictions);
    cudaMalloc((void**)&d_aux, sizeAux);

    FillingMatrices(h_predictions, 1, ny, nx);
    FillingMatrices(h_aux, 0, ny, 3);
    Predictions(h_targetValues, nx, 1);
    VectorVacio(h_accuracy, ny, 0);

    cudaMemcpy(d_accuracy, h_accuracy, sizeAccuracy, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targetValues, h_targetValues, sizeTargetValues, cudaMemcpyHostToDevice);
    cudaMemcpy(d_predictions, h_predictions, sizePredictions, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, h_aux , sizeAux, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    f1_score<<<grid, block>>>(d_targetValues, d_predictions, d_accuracy, nx, ny, d_aux);
    cudaDeviceSynchronize();

    cudaMemcpy(h_accuracy, d_accuracy, sizeAccuracy, cudaMemcpyDeviceToHost);

    PrintVect(h_accuracy, ny);

    cudaDeviceReset();
}
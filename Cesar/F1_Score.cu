#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

//TODO: Probar codigo con diferentes escenarios 
//TODO: Desarrollar mejor vesion en py del f1
//TODO: Limpiar codigo
//TODO: Optimizar el codigo

__global__ void F1_Score(float *y_true, float *y_pred, float *f1_score, int nx, int ny, unsigned int *aux)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    unsigned int cesar = iy * 2;
    unsigned int segunda = cesar + 1;
    unsigned int tercera = cesar + 2;
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        // printf("ix%i iy%i tid %i cesar %i, segunda %i, tercera %i\n", ix, iy, tid, cesar, segunda, tercera);
        if (y_pred[tid] == 1 && y_true[ix] == 1) // TP A
        {
            atomicAdd(&aux[cesar], 1);
        }
        if (y_pred[tid] == 0 && y_true[ix] == 1) // FN B
        {
            atomicAdd(&aux[segunda], 1);
        }
        if (y_pred[tid] == 1 && y_true[ix] == 0) // FP C
        {
            atomicAdd(&aux[tercera], 1);
        }
        if (tid == nx - 1)
        {
            unsigned int a = aux[cesar];
            unsigned int b = aux[segunda];
            unsigned int c = aux[tercera];
            unsigned int x = (a + 0.5 * (b + c));
            if (x == 0){
                f1_score[iy] = -1;
                printf("Warning: Zero divition in f1_score[%i], value was set to -1.\n", iy);
            }                
            else
                f1_score[iy] = a / x;
        }
    }
}
void FillingMatrices(float *matrix, float num, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            matrix[i * m + j] = num;
}
void FillingMatrices(int *matrix, int num, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int e = 0; e < m; e++)
            matrix[i * m + e] = num;
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
void PrintVect(float *vect, int ny)
{
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
    int ny = 1;
    int nx = 8;
    int nm = ny * nx;

    // Memory size
    int nBytesPredictions = nm * sizeof(float);
    int nBytesTargetValues = nx * sizeof(float);
    int nBytesAccuracy = ny * sizeof(float);
    int nBytesAux = ny * 3 * sizeof(unsigned int);

    // Host memory allocation
    float *predictions, *targetValues, *h_accuracy;
    int *h_aux;
    h_accuracy = (float *)malloc(nBytesAccuracy);
    h_aux = (int *)malloc(nBytesAux);
    cudaMallocHost((void **)&predictions, nBytesPredictions);
    cudaMallocHost((void **)&targetValues, nBytesTargetValues);

    // Device memory allocation
    float *d_accuracy;
    unsigned int *d_aux;
    cudaMalloc((void **)&d_accuracy, nBytesAccuracy);
    cudaMalloc((void **)&d_aux, nBytesAux);

    // Host memory initialization
    // y_true = [1, 0, 1, 1, 0, 1, 0, 1]
    // y_pred = [1, 0, 1, 0, 1, 1, 0, 1]
    predictions[0] = 1;
    predictions[1] = 0;
    predictions[2] = 0;
    predictions[3] = 0;
    predictions[4] = 0;
    predictions[5] = 0;
    predictions[6] = 0;
    predictions[7] = 0;
    // FillingMatrices(predictions, 1, ny, nx);
    FillingMatrices(h_aux, 0, ny, 3);
    // Predictions(targetValues, nx, 1);
    targetValues[0] = 0;
    targetValues[1] = 0;
    targetValues[2] = 1;
    targetValues[3] = 0;
    targetValues[4] = 0;
    targetValues[5] = 0;
    targetValues[6] = 0;
    targetValues[7] = 1;
    VectorVacio(h_accuracy, nx, 0);

    // Memory transfer host to device
    cudaMemcpy(d_accuracy, h_accuracy, nBytesAccuracy, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, h_aux, nBytesAux, cudaMemcpyHostToDevice);

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    F1_Score<<<grid, block>>>(targetValues, predictions, d_accuracy, nx, ny, d_aux);
    cudaDeviceSynchronize();

    // Memory transfer device to host
    cudaMemcpy(h_accuracy, d_accuracy, nBytesAccuracy, cudaMemcpyDeviceToHost);

    PrintVect(h_accuracy, ny);

    // Reset device
    cudaDeviceReset();
}
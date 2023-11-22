#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
// TODO: Desarrollar una mejor version en python para compararlos bien
// TODO: Documentar el codigo
// TODO: Probar un problema real con el codigo
// TODO: Optimizar rendimiento general

// Accuracy clasification score given a y_true and a semantic matrix
// Parameters:
//----------
// float *y_pred: Matriz de ny x nx dimensiones que representa las predicciones
// realizada por ny individuos con nx parametros.
// float *y_true: Array de nx elementos que representa los valores esperados de las
// predicciones realizadas por los individuos
// float *accuracy: Array de ny elementos que representa el accuracy de ny individuos
// int nx: nx elementos
// int ny: ny elementos
__global__ void accuracyScore(float *y_true, float *y_pred, float *accuracies, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int tid = iy * nx + ix;

    if (tid < nx * ny && ix < nx && iy < ny)
    {
        if (y_pred[tid] == y_true[ix])
        {
            float sum = 1 / (float)nx;
            atomicAdd(&accuracies[iy], sum);
        }
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
            vector[i] = 1;
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
    int ny = 2;
    int nx = 2;
    int nm = ny * nx;

    // Memory size
    int nBytesPredictions = nm * sizeof(float);
    int nBytesTargetValues = nx * sizeof(float);
    int nBytesAccuracy = ny * sizeof(float);

    // Host memory allocation
    float *predictions, *targetValues, *h_accuracy;

    h_accuracy = (float *)malloc(nBytesAccuracy);
    cudaMallocHost((void **)&predictions, nBytesPredictions);
    cudaMallocHost((void **)&targetValues, nBytesTargetValues);

    // Device memory allocation
    float *d_accuracy;
    cudaMalloc((void **)&d_accuracy, nBytesAccuracy);

    // Host memory initialization
    predictions[0] = 1;
    predictions[1] = 1;
    predictions[2] = 1;
    predictions[3] = 1;

    targetValues[0] = 1;
    targetValues[1] = 0;

    VectorVacio(h_accuracy, ny, 0);

    // Memory transfer host to device
    cudaMemcpy(d_accuracy, h_accuracy, nBytesAccuracy, cudaMemcpyHostToDevice);

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    accuracyScore<<<grid, block>>>(targetValues, predictions, d_accuracy, nx, ny);
    cudaDeviceSynchronize();

    // Memory transfer device to host
    cudaMemcpy(h_accuracy, d_accuracy, nBytesAccuracy, cudaMemcpyDeviceToHost);

    PrintVect(h_accuracy, ny);

    // Reset device
    cudaDeviceReset();
}
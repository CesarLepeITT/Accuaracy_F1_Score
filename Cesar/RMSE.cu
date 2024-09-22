#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "./Cesar.cpp"

int main()
{
    // Size
    int nx, ny;

    ny = 1;
    nx = 1;

    // Memory size
    int sizeYPred, sizeYTrue;
    sizeYPred = sizeof(float) * nx * ny;
    sizeYTrue = sizeof(float) * nx;

    // Device initialization
    float *y_pred, *yTrue;
    cudaMallocManaged((void **)&y_pred, sizeYPred);
    cudaMallocManaged((void **)&yTrue, sizeYTrue);

    printf("Ingrese el valor de x: ");
    scanf("%f", &y_pred[0]);

    printf("Ingrese el valor de y: ");
    scanf("%f", &yTrue[0]);

    // Kernell call
    RMSE<<<1, 1>>>(y_pred, yTrue, nx, ny);
    cudaDeviceSynchronize();

    // Device reset
    cudaDeviceReset();
}

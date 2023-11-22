#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

// TODO: Probar codigo con diferentes escenarios
// TODO: Desarrollar mejor vesion en py del f1
// TODO: Limpiar codigo
// TODO: Optimizar el codigo

__global__ void f1_score(float *y_pred, float *y_true, float *f1_scores, int nx, int ny, unsigned int *aux)
{
    // Calculate global coordinates of the thread in the 2D block
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // Calculate global index of the y_pred matrix
    unsigned int tid = iy * nx + ix;

    // Calculate positions in the auxiliary array for TP, FN, and FP
    unsigned int pos_sum_TP = iy * 3;
    unsigned int pos_sum_FN = pos_sum_TP + 1;
    unsigned int pos_sum_FP = pos_sum_TP + 2;

    // Check if the thread is within the data bounds
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        // Check conditions for TP, FN, and FP, and update partial sums atomically
        if (y_pred[tid] == 1 && y_true[ix] == 1) // TP
        {
            atomicAdd(&aux[pos_sum_TP], (float)1.0);
        }
        if (y_pred[tid] == 0 && y_true[ix] == 1) // FN
        {
            atomicAdd(&aux[pos_sum_FN], (float)1.0);
        }
        if (y_pred[tid] == 1 && y_true[ix] == 0) // FP
        {
            atomicAdd(&aux[pos_sum_FP], (float)1.0);
        }
        //  Check if it is the last thread in the row
        if (ix == nx - 1)
        {
            // Retrieve partial sums
            unsigned int TP = aux[pos_sum_TP];
            unsigned int FN = aux[pos_sum_FN];
            unsigned int FP = aux[pos_sum_FP];

            // Calculate the F1 score and store it in the f1_scores array
            float f1_denominator = (float)(TP + 0.5 * (FN + FP));
            if (f1_denominator == 0)
            {
                f1_scores[iy] = (float)-1; // Avoid division by zero
            }
            else
                f1_scores[iy] = (float)(TP / f1_denominator);

        }
    }
}
void FillingMatrices(float *matrix, float num, int n, int m)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            // Calcula el índice correspondiente en el arreglo unidimensional
            int index = i * m + j;
            matrix[index] = 1.0; // Llena la matriz con el valor 1.0
        }
    }
}
void FillingMatrices(unsigned int *matrix, int num, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int e = 0; e < m; e++)
            matrix[i * m + e] = num;
}
void Predictions(float *vector, int m, float num)
{
    for (int i = 0; i < m; i++)
    {
        vector[i] = num;
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
void CheckResults(float *matriz, int m)
{
    int aux = 0;

        for (int j = 0; j < m; ++j)
        {
            // Calcula el índice correspondiente en el arreglo unidimensional
            int index = j;
            if (matriz[index] != 1) // Llena la matriz con el valor 1.0
                aux++;
        }
    
    if (aux == 0)
        printf("TOdo bien\n");
    else
        printf("Algo salio mal %i\n", aux);
}

int main()
{
    // Set up dimensions
    int ny = 2048;
    int nx = 2048;
    int nm = ny * nx;

    // Memory size
    int nBytesPredictions = nm * sizeof(float);
    int nBytesTargetValues = nx * sizeof(float);
    int nBytesAccuracy = ny * sizeof(float);
    int nBytesAux = ny * 3 * sizeof(unsigned int);

    // Host memory allocation
    float *predictions, *targetValues, *h_accuracy;
    unsigned int *h_aux;
    h_accuracy = (float *)malloc(nBytesAccuracy);
    h_aux = (unsigned int *)malloc(nBytesAux);
    cudaMallocManaged((void **)&predictions, nBytesPredictions);
    cudaMallocManaged((void **)&targetValues, nBytesTargetValues);

    // Device memory allocation
    float *d_accuracy;
    unsigned int *d_aux;
    cudaMalloc((void **)&d_accuracy, nBytesAccuracy);
    cudaMalloc((void **)&d_aux, nBytesAux);

    // Host memory initialization

    FillingMatrices(predictions, 1, ny, nx);

    FillingMatrices(h_aux, 0, ny, 3);

    Predictions(targetValues, nx, 1);

    VectorVacio(h_accuracy, nx, 0);

    // Memory transfer host to device
    cudaMemcpy(d_accuracy, h_accuracy, nBytesAccuracy, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, h_aux, nBytesAux, cudaMemcpyHostToDevice);

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    f1_score<<<grid, block>>>(predictions, targetValues, d_accuracy, nx, ny, d_aux);
    cudaDeviceSynchronize();

    // Memory transfer device to host
    cudaMemcpy(h_accuracy, d_accuracy, nBytesAccuracy, cudaMemcpyDeviceToHost);

    // PrintVect(h_accuracy, ny);
    CheckResults(h_accuracy, ny);

    // Reset device
    cudaDeviceReset();
}
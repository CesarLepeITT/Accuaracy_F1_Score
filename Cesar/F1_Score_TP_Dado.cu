#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void f1_score(float *y_pred, float *y_true, float *cases_TP, float *f1_scores, int nx, int ny, unsigned int *aux)
{
    // Calculate global coordinates of the thread in the 2D block
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // Calculate global index of the y_pred matrix
    unsigned int tid = iy * nx + ix;

    // Calculate positions in the auxiliary array for TP, FN, and FP
    unsigned int pos_sum_TP = iy * 2;
    unsigned int pos_sum_FN_FP = pos_sum_TP + 1;

    // Check if the thread is within the data bounds
    if (tid < nx * ny && ix < nx && iy < ny)
    {
        int nElemTP = 3; //Corregir esto
        bool isTP = false;

        for (int i = 0; i < nElemTP; i++)
        {
            if (y_true[ix] == cases_TP[i])
            {
                isTP = true;
                printf("Entro\n case: %f, ytrue: %f ix: %i iy: %i\n", cases_TP[i], y_true[ix], ix, iy);
                i = nElemTP;
            }
            //printf("%i \n", i);
        }

        // Check conditions for TP, FN, and FP, and update partial sums atomically
        if (y_pred[tid] == y_true[ix] && isTP == true) // TP
        {
            printf("TP ix%i iy%i, pred %f y_true %f \n", ix, iy,y_pred[tid],y_true[ix]);
            atomicAdd(&aux[pos_sum_TP], 1.0F);
        }
        if (y_pred[tid] != y_true[ix]) // FN and FP (FN + FP)
        {
            printf("FN+Fp ix%i iy%i, pred %f y_true %f \n", ix, iy,y_pred[tid],y_true[ix]);
            atomicAdd(&aux[pos_sum_FN_FP], 1.0F);
        }

        //  Check if it is the last thread in the row
        if (ix == nx - 1)
        {
            // Retrieve partial sums
            unsigned int TP = aux[pos_sum_TP];
            unsigned int FN_FP = aux[pos_sum_FN_FP];

            // Calculate the F1 score and store it in the f1_scores array
            float f1_denominator = (float)(TP + 0.5F * FN_FP);
            if (f1_denominator == 0)
            {
                f1_scores[iy] = -1.0F; // Avoid division by zero
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
void imprimirMatriz(float *matriz, int filas, int columnas) {
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            printf("%f ", *(matriz + i * columnas + j));
        }
        printf("\n");
    }
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
    int ny = 2;
    int nx = 2;
    int nm = ny * nx;

    // Memory size
    int nBytesPredictions = nm * sizeof(float);
    int nBytesTargetValues = nx * sizeof(float);
    int nBytesAccuracy = ny * sizeof(float);
    int nBytesAux = ny * 2 * sizeof(unsigned int);

    // Host memory allocation
    float *predictions, *targetValues, *h_f1Scores, *cases_TP;
    unsigned int *h_aux;
    h_f1Scores = (float *)malloc(nBytesAccuracy);
    h_aux = (unsigned int *)malloc(nBytesAux);

    cudaMallocManaged((void **)&predictions, nBytesPredictions);
    cudaMallocManaged((void **)&targetValues, nBytesTargetValues);
    cudaMallocManaged((void **)&cases_TP, 3 * sizeof(float));

    // Device memory allocation
    float *d_f1Scores;
    unsigned int *d_aux;
    cudaMalloc((void **)&d_f1Scores, nBytesAccuracy);
    cudaMalloc((void **)&d_aux, nBytesAux);

    // Host memory initialization

    // FillingMatrices(predictions, 1, ny, nx);
    targetValues[0] = 0;
    targetValues[1] = 0;


    FillingMatrices(h_aux, 0, ny, 2);

    // Predictions(targetValues, nx, 1);

    predictions[0] = 0;
    predictions[1] = 0;
    predictions[2] = 2;
    predictions[3] = 2;

    VectorVacio(h_f1Scores, nx, 0);

    cases_TP[0] = 1;
    cases_TP[1] = 2;
    cases_TP[2] = 3;

    // Memory transfer host to device
    cudaMemcpy(d_f1Scores, h_f1Scores, nBytesAccuracy, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, h_aux, nBytesAux, cudaMemcpyHostToDevice);

    // Kernell call
    int dimx = 32;
    int dimy = 32;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    f1_score<<<grid, block>>>(predictions, targetValues, cases_TP, d_f1Scores, nx, ny, d_aux);
    // cudaDeviceSynchronize();
    cudaFree(d_aux);
    // Memory transfer device to host
    cudaMemcpy(h_f1Scores, d_f1Scores, nBytesAccuracy, cudaMemcpyDeviceToHost);

    printf("Target values:\n");
    imprimirMatriz(targetValues, ny, nx);
    printf("Predicciones:\n");
    imprimirMatriz(predictions, ny, nx);
    PrintVect(h_f1Scores, ny);
    // CheckResults(h_accuracy, ny);

    // Reset device
    cudaDeviceReset();
}
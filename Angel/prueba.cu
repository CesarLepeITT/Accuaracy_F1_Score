#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void accuracy_score(int* _true, int* y_pred, bool truE){
    
}

void fillingMatrices(int* matrix, int n, int m){
    for (int i = 0; i < n; i++)
        for (int e = 0; e < m; e++)
            matrix [i * m + e] = e;
}

void predictions(int* vector, int m){
    for (int i = 0; i < m; i++)
    {
        vector[i] = i;
    }
    
}

int main(){
    // predicciones y valores esperados 
    int* predictions_h, * predictions_d, * targValues_h, * targValues_d;
    bool ban;
    int n = 5;
    int m = 6;
    size_t pitch;
    predictions_h = (int *)malloc(n * m * sizeof(int));
    targValues_h = (int*)malloc(m * sizeof(int));
    fillingMatrices(predictions_h, n, m);
    for (int i = 0; i < n; i++){
        for (int e = 0; e < m; e++){
            printf("|%i|", predictions_h[i * m + e]);
        }
        printf("\n");
    }
    predictions(targValues_h, m);
    cudaMallocPitch((void**)&predictions_d, &pitch, m * sizeof(int), n);
    cudaMemcpy2D(predictions_d, pitch, predictions_h, m * sizeof(int), m * sizeof(int), n, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&targValues_d, m * sizeof(int));
    cudaMemcpy(targValues_d, targValues_h, m * sizeof(int), cudaMemcpyHostToDevice);

cudaDeviceReset();
}
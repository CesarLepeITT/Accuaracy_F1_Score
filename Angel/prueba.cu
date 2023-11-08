#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>


// cada hilo evalua un individuo y por ende cada hilo ocupa acceder a los datos reales (true values)
// por ello, almacenaremos por bloque una porcion de los datos del array con los true values en la compartida, cada hilo (dentro del bloque)
// podra acceder a ella para hacer sus calculos, comparando la data correspondiente del individuo que le fue asignado

__global__ void accuracy_score(int* _true, int* y_pred, bool truE){
    int idY = threadIdx.y;
    int idX = threadIdx.x;
    __shared__ int predS[32];


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

    // Matriz con 5 individuos y 6 columnas de datos
    int n = 5;
    int m = 6;



    size_t pitch;
    predictions_h = (int *)malloc(n * m * sizeof(int));
    targValues_h = (int*)malloc(m * sizeof(int)); // el tama;o del vector de valores esperados es del tama;o de columnas de datos
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
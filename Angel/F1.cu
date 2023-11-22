#include<stdlib.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include <unordered_set>
#include <unordered_map>
#include <random>

__global__ void kernel(int *p){
    p[threadIdx.x] = 1;
}

__global__ void F1(float* y_true, float* y_pred, int m, int noClasses, float* y_trueEachClass, float* TP, float* FP, float* FN){
    for (int i = 0; i < noClasses; i++){
        if(y_pred[threadIdx.x] == y_true[threadIdx.x] && y_pred[threadIdx.x] == y_trueEachClass[i]) { atomicAdd(&TP[i], 1); printf("TP[%i]++\n", i); }
        if(y_pred[threadIdx.x] != y_true[threadIdx.x] && y_pred[threadIdx.x] == y_trueEachClass[i]) { atomicAdd(&FP[i], 1); printf("FP[%i]++\n", i); }
        if(y_pred[threadIdx.x] != y_true[threadIdx.x] && y_true[threadIdx.x] == y_trueEachClass[i]) { atomicAdd(&FP[i], 1); printf("FN[%i]++\n", i); }
    }
            
            
    
}


void getNoClasses(float* y_true, int m, int& noClasses){
    std::unordered_set<int> elementosUnicos;
    for (int i = 0; i < m; i++)
        elementosUnicos.insert(y_true[i]);             // Esta parte obtiene el No. clases
    noClasses = elementosUnicos.size();
}

void getVector(float* vector, int size){
    for (int i = 0; i < size; i++){ 
        if(i % 2 == 0) vector[i] = 1;
        else vector[i] = 2;
    }
        
        
}

void setClasses(float* y_true, int m, int& noClasses, float* y_trueEachClass){
    int temp = y_true[0];                                                 // Esta declaracion de temp nos servira para almacenar el valor de cada clase
    for (int i = 0; i < noClasses; i++)
    {
        y_trueEachClass[i] = temp;                                        // Con este algoritmo conseguimos llenar el array con longiud [no Classes]
        for (int e = 0; e < m; e++)                                       // con el valor dado de cada clase (ej. clase 0 = 45, clase 1 = 32)
           if (y_true[e] == y_trueEachClass[i]) y_true[e] = 0; 
        for (int e = 0; e < m; e++)
            if (y_true[e] != 0) { temp = y_true[e]; e = m; }    
    }
}

int main(){

    // float* y_pred, * y_true, * y_trueEachClass;
    float y_pred[6] = {0, 2, 1, 0, 0, 1}; float y_true [6] = {0, 1, 2, 0, 1, 2}; float* y_trueEachClass;

    float* TP, * FP, * FN;
    float* y_pred_d, * y_true_d, *y_trueEachClass_d;
    float* TP_d, * FP_d, * FN_d;
    int m = 6;                                                           // Dimensiones del array
    int noClasses = 0;


    // y_true = (float*)malloc(m * sizeof(float));                           // Reservacion de memoria del array de los valores objetivos y de los predecidos
    // y_pred = (float*)malloc(m * sizeof(float));
    // getVector(y_true, m);                                                 // Inicializacion de memoria de dicho array
    // getVector(y_pred, m);  

    cudaMalloc((void**)&y_true_d, m * sizeof(float));
    cudaMalloc((void**)&y_pred_d, m * sizeof(float));


    cudaMemcpy(y_true_d, y_true, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_pred_d, y_pred, m * sizeof(float), cudaMemcpyHostToDevice);


    for (int i = 0; i < m; i++)
    {
        printf("[%f], ", y_true[i]);                                      // Para visualizar el array original
    }
    
    printf("\n\n");

    getNoClasses(y_true, m, noClasses);                                   // De esta funcion obtenemos: 1. No. clases, 2. vector con el valor dado de cada clase (ej. clase 0 = 45, clase 1 = 32)         
    printf("%i\n\n\n\n", noClasses);

    y_trueEachClass = (float*)malloc(noClasses * sizeof(float));          // Declaracion dinamica de un array con longitud [noClasses]
    TP = (float*)malloc(noClasses * sizeof(float));
    FP = (float*)malloc(noClasses * sizeof(float));
    FN = (float*)malloc(noClasses * sizeof(float));


    cudaMalloc((void**)&y_trueEachClass_d, noClasses * sizeof(float));    // Alojacion en device dinamica del array con longitud [noClases]
    cudaMalloc((void**)&TP_d, noClasses * sizeof(float));
    cudaMalloc((void**)&FP_d, noClasses * sizeof(float));
    cudaMalloc((void**)&FN_d, noClasses * sizeof(float));
    
    setClasses(y_true, m, noClasses, y_trueEachClass);                                                          // Obtenido el vector del tipo: clase 0 = 4, clase 1 = 9...
    cudaMemcpy(y_trueEachClass_d, y_trueEachClass, noClasses * sizeof(float), cudaMemcpyHostToDevice);          // Hacemos la transferencia de datos


    for (int i = 0; i < noClasses; i++)
    {
        printf("[%f], ", y_trueEachClass[i]);                             // Para visualizar el array nuevo
    }
    
    F1<<<1,m>>>(y_true_d, y_pred_d, m, noClasses, y_trueEachClass_d, TP_d, FP_d, FN);
    cudaDeviceSynchronize();
    
    cudaMemcpy(TP, TP_d, noClasses * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(FP, FP_d, noClasses * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(FN, FN_d, noClasses * sizeof(float), cudaMemcpyDeviceToHost); 

    float F1 = 0;
    for (int i = 0; i < noClasses; i++)
        F1 += (TP[i] / (TP[i] + ((FP[i] + FN[i]) / 2)));
    F1 /= noClasses;
    
    printf("F1 MACRO: %f, TP_1: %f, FP_1: %f, FN_1: %f, TP_2: %f, FP_2: %f, FN_2: %f, TP_3: %f, FP_3: %f, FN_3: %f", F1, TP[0], FP[0], FN[0], TP[1], FP[1], FN[1], TP[2], FP[2], FN[2]);
    

    cudaDeviceReset();

}
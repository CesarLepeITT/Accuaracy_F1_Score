#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void kernel(float* mSemantica, float* targetValues, float* accuracyScore, int m){
    if (mSemantica[threadIdx.y * m + threadIdx.x] == targetValues[threadIdx.x])
    {
        //printf("thread y mSemantica[%i][%i]: %f = target[%i]: %f\n", threadIdx.y, threadIdx.x, mSemantica[threadIdx.y * m + threadIdx.x], threadIdx.x, targetValues[threadIdx.x]);
        atomicAdd(&accuracyScore[threadIdx.y], 1);
    }
    
    if (threadIdx.x == m - 1){
        // if (accuracyScore[threadIdx.y] > 0) printf ("%f\n",accuracyScore[threadIdx.y] / m);
        printf("%i\n", threadIdx.y);
    }
        
    
}

__global__ void kernel1(){
}
void Llenarvector(float* vector, int n, int m, int value){
    for (int i = 0; i < n * m; i++)
        vector[i] = i;
}
void Llenarmatriz(float* vector, int n, int m){
    for (int i = 0; i < m * n ; i++)
        vector[i] = i; 
}

int main(){
    float* targetValues; float* mSemantica; float* accuracyScore;
    float* targetValues_d; float* mSemantica_d; float* accuracyScore_d;
    int n = 100;
    int m = 11;
    targetValues = (float*)malloc(m * sizeof(float));
    mSemantica = (float*)malloc(n * m * sizeof(float));
    accuracyScore = (float*)malloc(n * sizeof(float));

    Llenarmatriz(mSemantica, n, m);
    Llenarvector(targetValues, 1, m, 1);

    cudaMalloc((void**)&targetValues_d, m * sizeof(float));
    cudaMalloc((void**)&mSemantica_d, n * m * sizeof(float));
    cudaMalloc((void**)&accuracyScore_d, n* sizeof(float));

    cudaMemcpy(targetValues_d, targetValues, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mSemantica_d, mSemantica, n * m * sizeof(float), cudaMemcpyHostToDevice);


    dim3 block (m,n);
    kernel<<<1,block>>>(mSemantica_d, targetValues_d, accuracyScore_d, m);
    cudaDeviceSynchronize();

    cudaMemcpy(accuracyScore, accuracyScore_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    /*for (int i = 0; i < n; i++)
    {
        printf("Accuracy Score [%i]: %f\n", i, accuracyScore[i]);
    }*/
    


    cudaDeviceReset();
}
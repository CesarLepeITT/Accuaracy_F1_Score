#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void kernel(float* mSemantica, float* targetValues, float* accuracyScore, int n, int m){
    if (mSemantica[threadIdx.y * m + threadIdx.x] == targetValues[threadIdx.x])
    {
        printf("thread y mSemantica[%i][%i]: %f = target[%i]: %f", threadIdx.y, threadIdx.x, mSemantica[threadIdx.y * m + threadIdx.x], threadIdx.x, targetValues[threadIdx.x]);
    }
    printf("hola");
    
}

__global__ void kernel1(){
    printf("holaaaaaa");
}
void Llenarvector(float* vector, int n, int m){
    for (int i = 0; i < n * m; i++)
        vector[i] = 1;
}

int main(){
    float* targetValues; float* mSemantica; float* accuracyScore;
    float* targetValues_d; float* mSemantica_d; float* accuracyScore_d;
    int n = 2;
    int m = 5;
    targetValues = (float*)malloc(m * sizeof(float));
    mSemantica = (float*)malloc(n * m * sizeof(float));
    accuracyScore = (float*)malloc(n * sizeof(float));

    Llenarvector(mSemantica, n, m);
    Llenarvector(targetValues, 1, m);

    cudaMalloc((void**)&targetValues_d, m * sizeof(float));
    cudaMalloc((void**)&mSemantica, n * m * sizeof(float));
    cudaMalloc((void**)&accuracyScore_d, n* sizeof(float));

    cudaMemcpy(targetValues_d, targetValues, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mSemantica_d, mSemantica, n * m * sizeof(float), cudaMemcpyHostToDevice);


    dim3 block (m,n);
    kernel<<<1,block>>>(mSemantica, targetValues_d, accuracyScore_d, n, m);
    cudaDeviceSynchronize();
    kernel1<<<1,1>>>();
    cudaDeviceReset();
}
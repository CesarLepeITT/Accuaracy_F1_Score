#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda_runtime.h>

__global__ void RMSE(float* yhat, float* y, int n, float* sum){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
        atomicAdd(sum, pow(yhat[idx] - y[idx], 2));
    if (idx == 0){
        *sum = sqrt(*sum / n);
        printf("RMSE: %f", *sum);
    } 
}

void llenarVector(float *vector, int n){
    for (int i = 0; i < n; i++)
        vector[i] = 1;
}

void llenarVectorPred(float *vector, int n){
    for (int i = 0; i < n; i++)
        vector[i] = i;
}


int main(){
    float* yhat_h, *yhat_d ;
    float* y_h, *y_d;
    float* sum_h, *sum_d;
    int n = 5;

    yhat_h = (float*)malloc(n * sizeof(float));
    y_h = (float*)malloc(n * sizeof(float));
    sum_h = (float*)malloc(sizeof(float));

    llenarVector(yhat_h, n);
    llenarVectorPred(y_h, n);

    cudaMalloc((void**)&yhat_d, n * sizeof(float));
    cudaMalloc((void**)&y_d, n * sizeof(float));
    cudaMalloc((void**)&sum_d, sizeof(float));

    cudaMemcpy(yhat_d, yhat_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block (32);
    dim3 grid((n + block.x - 1) / block.x);
    RMSE<<<grid,block>>>(yhat_d,y_d, n, sum_d);
    cudaDeviceSynchronize();
    cudaMemcpy(sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nRMSE (host) = %f\n", *sum_h);
    cudaDeviceReset();
}
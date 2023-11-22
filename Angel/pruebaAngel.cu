#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void kernel(float* mSemantica, float* targetValues, float* accuracyScore){
    if (mSemantica[((blockDim.y * blockIdx.y + threadIdx.y) * (gridDim.x * blockDim.x)) + (blockDim.x * blockIdx.x + threadIdx.x)] == targetValues[blockDim.x * blockIdx.x + threadIdx.x])
    {
        // printf("individuo: %i, thread y mSemantica[%i]: %f = target[%i]: %f\n", blockDim.y * blockIdx.y + threadIdx.y, ((blockDim.y * blockIdx.y + threadIdx.y) * (gridDim.x * blockDim.x)) + (blockDim.x * blockIdx.x + threadIdx.x), mSemantica[((blockDim.y * blockIdx.y + threadIdx.y) * (gridDim.x * blockDim.x)) + (blockDim.x * blockIdx.x + threadIdx.x)],blockDim.x * blockIdx.x +threadIdx.x, targetValues[blockDim.x * blockIdx.x + threadIdx.x]);
        atomicAdd(&accuracyScore[blockDim.y * blockIdx.y + threadIdx.y], 1);
    }
    __syncthreads();
    if (blockDim.x * blockIdx.x + threadIdx.x == m - 1){
        // printf("individuo: %i, accuracyscore: %f\n", blockDim.y * blockIdx.y + threadIdx.y, accuracyScore[blockDim.y * blockIdx.y + threadIdx.y]);
        
    }
}

__global__ void leer(float* mSemantica, float* targetValues, float* accuracyScore, int m){
    printf("thread: %i, y: %i\n",((blockDim.y * blockIdx.y + threadIdx.y) * (gridDim.x * blockDim.x)) + (blockDim.x * blockIdx.x + threadIdx.x),threadIdx.y);
     

}
void llenarVector(float* vector, int n, int m, int value){
    for (int i = 0; i < m; i++)
        vector[i] = 1;
}
void llenarMatriz(float* vector, int n, int m){
    for (int i = 0; i < n; i++){
        for (int e = 0; e < m; e++)
        {
            if (i % 2 == 0) { vector[i * m + e] = 1; }
            else {vector[i * m + e] = 0; }
        }
    }  
}



void splitM(int y, int x, int& newY, int& newX, int& gridY,  int& gridX){
    bool ok = true;
    bool ok2 = true;
    bool ok3 = true;
    int multiploY = 1;
    int multiploX = x;
    while (ok){
        if (x > pow (2, 31) - 1  && ok2){
            printf("hola");
            while (ok3){
                multiploX--;
                if ((x / multiploX) < pow (2, 31) - 1 && x % multiploX == 0) { ok = false; ok2 = false; }
            }
        }
        if (((y / multiploY) * (x / multiploX)) < 1024 && (y % multiploY) == 0 ) { ok = false; }
        else multiploY++;
    }
    newY = y / multiploY;
    newX = x / multiploX;
    gridY = (y + newY - 1) / newY;
    gridX = (x + newX - 1) / newX;

}


int main(){
    float* targetValues; float* mSemantica; float* accuracyScore;
    float* targetValues_d; float* mSemantica_d; float* accuracyScore_d;
    int n = 1024;
    int m = 1024;
    targetValues = (float*)malloc(m * sizeof(float));
    mSemantica = (float*)malloc(n * m * sizeof(float));
    accuracyScore = (float*)malloc(n * sizeof(float));

    llenarMatriz(mSemantica, n, m);
    llenarVector(targetValues, 1, m, 1);

    cudaMalloc((void**)&targetValues_d, m * sizeof(float));
    cudaMalloc((void**)&mSemantica_d, n * m * sizeof(float));
    cudaMalloc((void**)&accuracyScore_d, n* sizeof(float));

    cudaMemcpy(targetValues_d, targetValues, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mSemantica_d, mSemantica, n * m * sizeof(float), cudaMemcpyHostToDevice);

    int BdimY = 0;
    int BdimX = 0;

    int GdimY = 0;
    int GdimX = 0;


    splitM(n, m, BdimY, BdimX, GdimY, GdimX);
    dim3 block (BdimX,BdimY);
    dim3 grid(GdimX,GdimY);
    kernel<<<grid,block>>>(mSemantica_d, targetValues_d, accuracyScore_d, m);
    //leer<<<grid,block>>>(mSemantica_d, targetValues_d, accuracyScore_d, m);
    cudaDeviceSynchronize();
    printf("%i, %i, %i\n",block.y, block.x, grid.x);
    cudaMemcpy(accuracyScore, accuracyScore_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++)
    {
        printf("Accuracy Score [%i]: %f\n", i, accuracyScore[i] / m);
    }
    cudaDeviceReset();
}
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>


// cada hilo evalua un individuo y por ende cada hilo ocupa acceder a los datos reales (true values)
// por ello, almacenaremos por bloque una porcion de los datos del array con los true values en la compartida, cada hilo (dentro del bloque)
// podra acceder a ella para hacer sus calculos, comparando la data correspondiente del individuo que le fue asignado

__global__ void accuracy_score(int* _true, int* y_pred, bool truE, int m){
    __shared__ int trueS[16];
        int i = m / 16; // calculamos las veces que habra que almacenar espacios de 16 en la memoria compartida
        int e = m - i * 16; // calculamos lo que sobra de del multiplo anterior para al final hacer solo almacenar esa cantidad
        int mThreads = 16; // los threads necesarios en un inicio son 16
        for (int guardado = 0; guardado < i; guardado++)
        {
            if (threadIdx.x < mThreads){
                trueS [threadIdx.x] = _true[threadIdx.x + guardado * 16]; // los hilos necesarios copian los datos del vector de la data esperada (_true)
                printf("[%i]\n",trueS[threadIdx.x]);
            }
            if (guardado == i - 1 && e != 0) { i++; mThreads = e; e = 0;} // si existe sobra, hacemos que exista una iteracion mas y ademas cambiamos los threads necesarios y hacemos 'e = 0' para que no se cicle 
            __syncthreads();
                                                                  
        }
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

    // Matriz con 5 individuos y 6 columnas de datos (el array de valores esperados es de 6 elementos)
    int n = 5;
    int m = 35;



    size_t pitch;
    predictions_h = (int *)malloc(n * m * sizeof(int));
    targValues_h = (int*)malloc(m * sizeof(int)); // el tama;o del vector de valores esperados es del tama;o de columnas de datos
    fillingMatrices(predictions_h, n, m);
    //for (int i = 0; i < n; i++){
        //for (int e = 0; e < m; e++){
            //printf("|%i|", predictions_h[i * m + e]);
        //}
        //printf("\n");
    //}


    predictions(targValues_h, m);
    cudaMallocPitch((void**)&predictions_d, &pitch, m * sizeof(int), n);
    cudaMemcpy2D(predictions_d, pitch, predictions_h, m * sizeof(int), m * sizeof(int), n, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&targValues_d, m * sizeof(int));
    cudaMemcpy(targValues_d, targValues_h, m * sizeof(int), cudaMemcpyHostToDevice);
    accuracy_score<<<1,32>>>(targValues_d, predictions_d, ban, m);

cudaDeviceReset();
}
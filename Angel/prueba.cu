#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void prueba(){
    printf("hola");
}

int main(){

prueba<<<1,100>>> ();
cudaDeviceReset();
}
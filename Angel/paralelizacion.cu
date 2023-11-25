#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <unordered_set>
#include <cuda_pipeline_primitives.h>

__global__ void getNoClassesKernel(float* y_true, int m, int& noClasses){
    
}

void getNoClasses(float* y_true, int m, int& noClasses){
    std::unordered_set<int> elementosUnicos;
    for (int i = 0; i < m; i++)
        elementosUnicos.insert(y_true[i]);             // Esta parte obtiene el No. clases
    noClasses = elementosUnicos.size();
}

int main(){

}
#include <cuda_runtime.h>

//TODO
__global__ void AccuaracyScore(float*, float*); 
//TODO
__global__ void AccuaracyScore(float*, float*, bool);
//TODO
__global__ void AccuaracyScore(float*, float*, bool, float*);
//TODO
__global__ void AccuaracyScore(float*, float*, float*);
//TODO
void ClassificationReport(float*, float* /*, TargetNames*/ );

// El y_true siempre sera un vector 
// El y_pred sera o un vector o una matriz
// cada columna del y_pred corresponde a una row del y_true

int main()
{

}
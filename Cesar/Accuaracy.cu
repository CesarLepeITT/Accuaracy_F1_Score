#include <cuda_runtime.h>

// Headers CUDA

//TODO: Si solo se manda y_true y y_pred
__global__ void AccuaracyScore(float*, float*); 
//TODO: Si solo se manda y_true y y_pred y normalize (pro defecto es TRUE)
__global__ void AccuaracyScore(float*, float*, bool);
//TODO: Si solo se manda y_true y y_pred, normalize (pro defecto es TRUE) y sample weight (array)
__global__ void AccuaracyScore(float*, float*, bool, float*);
//TODO: Si solo se manda y_true y y_pred y sample weight
__global__ void AccuaracyScore(float*, float*, float*);

// Headers C++
//TODO: Hace un reporte de la siguiente forma:
//               precision    recall  f1-score   support
//
//     class 0       0.67      1.00      0.80         2
//     class 1       0.00      0.00      0.00         1
//     class 2       1.00      0.50      0.67         2
//
//    accuracy                           0.60         5
//   macro avg       0.56      0.50      0.49         5
//weighted avg       0.67      0.60      0.59         5
//
void ClassificationReport(float*, float* /*, TargetNames*/ );

// El y_true siempre sera un vector 
// El y_pred sera o un vector o una matriz
// cada columna del y_pred corresponde a una row del y_true

int main()
{

}

__global__ void AccuaracyScore(float *yTrue, float *yPred)
{

}
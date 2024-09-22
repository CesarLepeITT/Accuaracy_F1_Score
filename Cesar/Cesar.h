
__global__ void ActivationFunction(float *semantica, float alpha, int nx, int ny);

__global__ void RMSE(float *semantica, float *yTrue, int nx, int ny);

__global__ void accuracyScore(float *y_pred, float *y_true,  float *accuracies, int nx, int ny);


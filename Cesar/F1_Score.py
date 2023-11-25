from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def calcular_f1_score(y_pred, y_true):
    # Inicializar un vector para almacenar los accuracies
    accuracies = []

    # Calcular el accuracy para cada fila
    for i in range(len(y_pred)):
        accuracy = f1_score(y_true, y_pred, average='macro')
        accuracies.append(accuracy)

    return np.array(accuracies)

# Set up dimensions
ny = 2
nx = 2

# Inicializar memoria
y_true = [2,3,0,2] #rellenar_vector_rand(nx)
y_pred = [2,3,1,2]#rellenar_matriz_rand(ny, nx)

# Calcular el accuracy
macro = f1_score(y_true, y_pred, average='macro')#calcular_f1_score(y_pred, y_true)
micro = f1_score(y_true, y_pred, average='micro')#calcular_f1_score(y_pred, y_true)
weighted = f1_score(y_true, y_pred, average='weighted')#calcular_f1_score(y_pred, y_true)
accuracy = accuracy_score(y_true, y_pred)

# Imprimir resultados
print("y_pred: \n", y_pred)
print("y_true: \n", y_true)
print("F1 score macro: \n", macro)
print("F1 score micro / accuracy: \n", micro)
print("F1 score weighted: \n", weighted)




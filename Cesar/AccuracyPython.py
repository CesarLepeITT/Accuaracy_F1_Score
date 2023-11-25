from sklearn.metrics import accuracy_score
import numpy as np

def rellenar_matriz(ny, nx, valor):
    matriz = np.full((ny, nx), valor, dtype=float)
    return matriz

def rellenar_matriz_rand(ny, nx):
    matriz = np.random.randint(2, size=(ny, nx))
    return matriz

def rellenar_vector(nx, valor):
    vector = np.full(nx, valor, dtype=float)
    return vector

def rellenar_vector_rand(nx):
    vector = np.random.randint(2, size=nx)
    return vector

def calcular_accuracy_por_fila(matriz_predicciones, vector_etiquetas):
    # Inicializar un vector para almacenar los accuracies
    accuracies = []

    # Calcular el accuracy para cada fila
    for i in range(len(matriz_predicciones)):
        accuracy = accuracy_score(vector_etiquetas, matriz_predicciones[i])
        accuracies.append(accuracy)

    return np.array(accuracies)

# Set up dimensions
ny = 1
nx = 8

# Inicializar memoria

y_trues = [1,3,2,3,1,2,3,1] #rellenar_vector_rand(nx)
y_preds = [3,1,2,1,2,3,2,1]#rellenar_matriz_rand(ny, nx)

# Calcular el accuracy
accuracies = accuracy_score(y_trues,y_preds, normalize= True)#calcular_accuracy_por_fila(y_pred, y_true)

# Imprimir resultados
print("y_pred: \n", y_preds)
print("y_true: \n", y_trues)
print("Accuracies: \n", accuracies)


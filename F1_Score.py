from sklearn.metrics import f1_score
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

def calcular_f1_score(y_pred, y_true):
    # Inicializar un vector para almacenar los accuracies
    accuracies = []

    # Calcular el accuracy para cada fila
    for i in range(len(y_pred)):
        accuracy = f1_score(y_true, y_pred[i])
        accuracies.append(accuracy)

    return np.array(accuracies)

# Set up dimensions
ny = 2
nx = 2

# Inicializar memoria
y_true = rellenar_vector_rand(nx)
y_pred = rellenar_matriz_rand(ny, nx)

# Calcular el accuracy
accuracies = calcular_f1_score(y_pred, y_true)

# Imprimir resultados
print("y_pred: \n", y_pred)
print("y_true: \n", y_true)
print("F1 score: \n", accuracies)


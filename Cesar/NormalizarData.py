from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Datos de ejemplo con dos características
data = np.array([[111.0, 2.0],
                 [2.0, 4.0],
                 [3.0, 6.0]])

# Inicializar el escalador Min-Max
scaler = MinMaxScaler()

# Ajustar y transformar los datos
normalized_data = scaler.fit_transform(data)

# Imprimir los datos normalizados
print("Datos originales:\n", data)
print("\nDatos normalizados:\n", normalized_data)

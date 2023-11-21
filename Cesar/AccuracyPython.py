from sklearn.metrics import accuracy_score

# Supongamos que 'y_true' es la matriz de etiquetas reales y 'y_pred' es la matriz de predicciones.
# Puedes reemplazar estas matrices con tus propios datos.

y_true = [0,0,1,0,0,0,1,0]  # Ejemplo de etiquetas reales
y_pred = [0,0,1,0,0,0,0,0]  # Ejemplo de predicciones

# Calcular el accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')
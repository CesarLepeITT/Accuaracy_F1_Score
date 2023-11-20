from sklearn.metrics import f1_score

# Vectores de ejemplo (etiquetas reales y predicciones)
y_true = [1,1,1,1,1,1,1,1]
y_pred = [1, 0, 1, 0, 1, 1, 0, 1]

# Calcular el puntaje F1
f1 = f1_score(y_true, y_pred)

print("Puntaje F1:", f1)

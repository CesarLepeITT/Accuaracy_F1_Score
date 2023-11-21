from sklearn.metrics import f1_score

# Vectores de ejemplo (etiquetas reales y predicciones)
y_true = [0,0,0,0,0,0,0,1]
y_pred = [1,0,0,0,0,0,0,0]

# Calcular el puntaje F1
f1 = f1_score(y_true, y_pred, zero_division="warn")

print("Puntaje F1:", f1)

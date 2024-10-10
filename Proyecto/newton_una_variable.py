import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from mpl_toolkits.mplot3d import Axes3D

# 1. Cargar el dataset con el separador correcto
df = pd.read_csv(r'dataset_desercion_estudiantil.csv', sep=';')

# Preparamos los datos (solo Horas como característica)
X = df[['Horas_estudio_por_semana']]  # Solo la característica de Horas
y = df['Desercion']  # Etiqueta (0 o 1)

# Añadimos la columna de unos (intercepto) a X
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Convertimos y a un vector columna
y = y.values.reshape(-1, 1)

# 2. Definir la función de costo logístico
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1/m) * np.sum(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))

# 3. Implementar Newton-Raphson
def newton_raphson(X, y, iterations=10):
    m, n = X.shape
    theta = np.zeros((n, 1))  # Inicializamos los parámetros a cero
    
    for i in range(iterations):
        # Predicción
        h = sigmoid(X.dot(theta))
        
        # Gradiente (Primera derivada)
        gradient = X.T.dot(h - y)
        
        # Matriz Hessiana (Segunda derivada)
        H = (X.T.dot(np.diag((h * (1 - h)).reshape(-1))).dot(X))
        
        # Actualización de los parámetros
        theta -= np.linalg.inv(H).dot(gradient)
        
        # Mostrar el costo en cada iteración
        print(f"Iteración {i+1}: Costo = {log_likelihood(X, y, theta)}")
    
    return theta

# 5. Entrenar el modelo con Newton-Raphson (solo con Promedio_estudiantil)
theta_final = newton_raphson(X, y, iterations=10)

# 6. Evaluar el modelo con probabilidades
def predict_prob(X, theta):
    return sigmoid(X.dot(theta))  # Esto devuelve la probabilidad

# Predicciones con probabilidades
probabilities = predict_prob(X, theta_final).flatten()
y = y.flatten()

# 7. Curva ROC
fpr, tpr, thresholds = roc_curve(y, probabilities)
roc_auc = roc_auc_score(y, probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('Curva ROC para Deserción de Estudiantes (Horas de Estudio)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')
plt.show()

# 8. Histograma de probabilidades
plt.figure(figsize=(10, 6))
plt.hist(probabilities[y == 0], bins=20, alpha=0.5, label="No Deserta", color="blue")
plt.hist(probabilities[y == 1], bins=20, alpha=0.5, label="Deserta", color="red")
plt.title("Distribución de Probabilidades de Deserción (Horas de Estudio)")
plt.xlabel("Probabilidad de Deserción")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# 9. Calcular la precisión del modelo
def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

predictions = predict(X, theta_final)
accuracy = np.mean(predictions == y)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Cargar el dataset con el separador correcto
df = pd.read_csv(r'dataset_desercion_estudiantil.csv', sep=';')

# Preparamos los datos (features y etiquetas)
X = df[['Materias_inscritas', 'Promedio_estudiantil', 'Horas_estudio_por_semana', 'Edad']]  # Características relevantes
y = df['Desercion']  # Etiqueta (0 o 1)

# Añadimos la columna de unos (intercepto) a X
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Convertimos y a un array de NumPy y lo ajustamos en un vector columna
y = y.to_numpy().reshape(-1, 1)

# 2. Definir la función de costo logístico
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1/m) * np.sum(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))

# 3. Implementar el Método de Punto Fijo
def punto_fijo(X, y, iterations=10, tol=1e-5, learning_rate=0.01):
    m, n = X.shape
    theta = np.zeros((n, 1))  # Inicialización en cero
    
    for i in range(iterations):
        # Predicción
        h = sigmoid(X.dot(theta))
        
        # Gradiente (Primera derivada)
        gradient = X.T.dot(h - y)
        
        # Actualización de los parámetros
        theta_new = theta - learning_rate * gradient
        
        # Verificar convergencia
        if np.linalg.norm(theta_new - theta) < tol:
            print(f"Convergió en la iteración {i+1}")
            break
        
        # Actualizamos los parámetros
        theta = theta_new
        
        # Mostrar el costo en cada iteración
        print(f"Iteración {i+1}: Costo = {log_likelihood(X, y, theta)}")
    
    return theta

# 5. Entrenar el modelo con el Método de Punto Fijo
theta_punto_fijo = punto_fijo(X, y, iterations=10)

# 6. Evaluar el modelo con probabilidades usando el método de punto fijo
def predict_prob(X, theta):
    return sigmoid(X.dot(theta))  # Esto devuelve la probabilidad

# Predicciones con probabilidades usando el método de punto fijo
probabilities_punto_fijo = predict_prob(X, theta_punto_fijo).flatten()
y = y.flatten()

# 7. Curva ROC para el método de punto fijo
fpr, tpr, thresholds = roc_curve(y, probabilities_punto_fijo)
roc_auc = roc_auc_score(y, probabilities_punto_fijo)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('Curva ROC para Deserción de Estudiantes (Método de Punto Fijo)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')
plt.show()

# 8. Histograma de probabilidades para el método de punto fijo
plt.figure(figsize=(10, 6))
plt.hist(probabilities_punto_fijo[y == 0], bins=20, alpha=0.5, label="No Deserta", color="blue")
plt.hist(probabilities_punto_fijo[y == 1], bins=20, alpha=0.5, label="Deserta", color="red")
plt.title("Distribución de Probabilidades de Deserción (Método de Punto Fijo)")
plt.xlabel("Probabilidad de Deserción")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# 9. Calcular la precisión del modelo (Punto Fijo)
def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

predictions_punto_fijo = predict(X, theta_punto_fijo)
accuracy_punto_fijo = np.mean(predictions_punto_fijo == y)
print(f"Precisión del modelo con el Método de Punto Fijo: {accuracy_punto_fijo * 100:.2f}%")

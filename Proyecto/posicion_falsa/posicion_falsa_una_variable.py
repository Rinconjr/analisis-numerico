import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Cargar el dataset con el separador correcto
df = pd.read_csv(r'../dataset_desercion_estudiantil.csv', sep=';')

# Preparamos los datos (solo "Horas_estudio_por_semana" como característica)
X = df[['Horas_estudio_por_semana']]  # Solo usamos la variable Horas de estudio
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

# 3. Implementar el Método de la Posición Falsa
def posicion_falsa(X, y, iterations=10, tol=1e-5):
    m, n = X.shape
    # Inicializamos dos puntos iniciales
    theta_0 = np.zeros((n, 1))  # Inicialización en cero
    theta_1 = np.ones((n, 1))  # Inicialización en uno
    
    for i in range(iterations):
        # Predicción en ambos puntos
        h_0 = sigmoid(X.dot(theta_0))
        h_1 = sigmoid(X.dot(theta_1))
        
        # Gradiente en ambos puntos
        gradient_0 = X.T.dot(h_0 - y)
        gradient_1 = X.T.dot(h_1 - y)
        
        # Método de la posición falsa
        theta_new = theta_1 - (theta_1 - theta_0) * gradient_1 / (gradient_1 - gradient_0 + 1e-10)
        
        # Verificamos la convergencia
        if np.linalg.norm(theta_new - theta_1) < tol:
            print(f"Convergió en la iteración {i+1}")
            break
        
        # Actualizamos los puntos para la siguiente iteración
        theta_0, theta_1 = theta_1, theta_new
        
        # Mostrar el costo en cada iteración
        print(f"Iteración {i+1}: Costo = {log_likelihood(X, y, theta_1)}")
    
    return theta_1

# 5. Entrenar el modelo con el Método de la Posición Falsa usando solo "Horas_estudio_por_semana"
theta_posicion_falsa = posicion_falsa(X, y, iterations=10)

# 6. Evaluar el modelo con probabilidades usando el método de la posición falsa
def predict_prob(X, theta):
    return sigmoid(X.dot(theta))  # Esto devuelve la probabilidad

# Predicciones con probabilidades usando el método de la posición falsa
probabilities_posicion_falsa = predict_prob(X, theta_posicion_falsa).flatten()
y = y.flatten()

# 7. Curva ROC para el método de la posición falsa
fpr, tpr, thresholds = roc_curve(y, probabilities_posicion_falsa)
roc_auc = roc_auc_score(y, probabilities_posicion_falsa)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('Curva ROC para Deserción de Estudiantes (Horas de Estudio - Posición Falsa)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')
plt.show()

# 8. Histograma de probabilidades para el método de la posición falsa
plt.figure(figsize=(10, 6))
plt.hist(probabilities_posicion_falsa[y == 0], bins=20, alpha=0.5, label="No Deserta", color="blue")
plt.hist(probabilities_posicion_falsa[y == 1], bins=20, alpha=0.5, label="Deserta", color="red")
plt.title("Distribución de Probabilidades de Deserción (Horas de Estudio - Posición Falsa)")
plt.xlabel("Probabilidad de Deserción")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# 9. Calcular la precisión del modelo (Posición Falsa)
def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

predictions_posicion_falsa = predict(X, theta_posicion_falsa)
accuracy_posicion_falsa = np.mean(predictions_posicion_falsa == y)
print(f"Precisión del modelo con el Método de la Posición Falsa: {accuracy_posicion_falsa * 100:.2f}%")

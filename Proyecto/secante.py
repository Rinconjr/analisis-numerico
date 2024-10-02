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

# Convertimos y a un vector columna
y = y.values.reshape(-1, 1)

# 2. Definir la función de costo logístico
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1/m) * np.sum(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))

# 3. Implementar el Método de la Secante
def secante(X, y, iterations=10, tol=1e-5):
    m, n = X.shape
    # Inicializamos dos puntos iniciales para el método de la secante
    theta_0 = np.zeros((n, 1))  # Inicialización en cero
    theta_1 = np.ones((n, 1))  # Inicialización en uno
    
    for i in range(iterations):
        # Predicción en ambos puntos
        h_0 = sigmoid(X.dot(theta_0))
        h_1 = sigmoid(X.dot(theta_1))
        
        # Gradiente en ambos puntos
        gradient_0 = X.T.dot(h_0 - y)
        gradient_1 = X.T.dot(h_1 - y)
        
        # Diferencia en los gradientes y en los parámetros
        diff_grad = gradient_1 - gradient_0
        diff_theta = theta_1 - theta_0
        
        if np.linalg.norm(diff_grad) < tol:  # Condición de convergencia
            print(f"Convergió en la iteración {i+1}")
            break
        
        # Actualización de los parámetros usando el método de la secante
        # Evitamos la inversa directamente
        theta_new = theta_1 - (diff_theta / diff_grad) * gradient_1
        
        # Actualizamos los valores para la siguiente iteración
        theta_0, theta_1 = theta_1, theta_new
        
        # Mostrar el costo en cada iteración
        print(f"Iteración {i+1}: Costo = {log_likelihood(X, y, theta_1)}")
    
    return theta_1

# 5. Entrenar el modelo con el Método de la Secante
theta_secante = secante(X, y, iterations=10)

# 6. Evaluar el modelo con probabilidades usando el método de la secante
def predict_prob(X, theta):
    return sigmoid(X.dot(theta))  # Esto devuelve la probabilidad

# Predicciones con probabilidades usando el método de la secante
probabilities_secante = predict_prob(X, theta_secante).flatten()
y = y.flatten()

# 7. Curva ROC para el método de la secante
fpr, tpr, thresholds = roc_curve(y, probabilities_secante)
roc_auc = roc_auc_score(y, probabilities_secante)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('Curva ROC para Deserción de Estudiantes (Método de la Secante)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')
plt.show()

# 8. Histograma de probabilidades para el método de la secante
plt.figure(figsize=(10, 6))
plt.hist(probabilities_secante[y == 0], bins=20, alpha=0.5, label="No Deserta", color="blue")
plt.hist(probabilities_secante[y == 1], bins=20, alpha=0.5, label="Deserta", color="red")
plt.title("Distribución de Probabilidades de Deserción (Método de la Secante)")
plt.xlabel("Probabilidad de Deserción")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# 9. Calcular la precisión del modelo (Secante)
def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

predictions_secante = predict(X, theta_secante)
accuracy_secante = np.mean(predictions_secante == y)
print(f"Precisión del modelo con el Método de la Secante: {accuracy_secante * 100:.2f}%")

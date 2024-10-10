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

# Convertimos y a un vector columna
y = y.values.reshape(-1, 1)

# 2. Definir la función de costo logístico
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1/m) * np.sum(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))

# 3. Implementar el Método de la Aproximación de Weierstrass
def weierstrass(X, y, iterations=10, tol=1e-5):
    m, n = X.shape
    # Inicializamos los parámetros de Weierstrass
    theta = np.zeros((n, 1))  # Inicialización en cero
    
    for i in range(iterations):
        # Predicción usando los parámetros actuales
        h = sigmoid(X.dot(theta))
        
        # Gradiente (Primera derivada)
        gradient = X.T.dot(h - y)
        
        # Aproximación de Weierstrass
        theta_new = theta - gradient / (1 + np.abs(gradient))
        
        # Verificamos la convergencia
        if np.linalg.norm(theta_new - theta) < tol:
            print(f"Convergió en la iteración {i+1}")
            break
        
        # Actualizamos los parámetros
        theta = theta_new
        
        # Mostrar el costo en cada iteración
        print(f"Iteración {i+1}: Costo = {log_likelihood(X, y, theta)}")
    
    return theta

# 5. Entrenar el modelo con el Método de Weierstrass usando solo "Horas_estudio_por_semana"
theta_weierstrass = weierstrass(X, y, iterations=10)

# 6. Evaluar el modelo con probabilidades usando el método de Weierstrass
def predict_prob(X, theta):
    return sigmoid(X.dot(theta))  # Esto devuelve la probabilidad

# Predicciones con probabilidades usando el método de Weierstrass
probabilities_weierstrass = predict_prob(X, theta_weierstrass).flatten()
y = y.flatten()

# 7. Curva ROC para el método de Weierstrass
fpr, tpr, thresholds = roc_curve(y, probabilities_weierstrass)
roc_auc = roc_auc_score(y, probabilities_weierstrass)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('Curva ROC para Deserción de Estudiantes (Horas de Estudio - Método de Weierstrass)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')
plt.show()

# 8. Histograma de probabilidades para el método de Weierstrass
plt.figure(figsize=(10, 6))
plt.hist(probabilities_weierstrass[y == 0], bins=20, alpha=0.5, label="No Deserta", color="blue")
plt.hist(probabilities_weierstrass[y == 1], bins=20, alpha=0.5, label="Deserta", color="red")
plt.title("Distribución de Probabilidades de Deserción (Horas de Estudio - Método de Weierstrass)")
plt.xlabel("Probabilidad de Deserción")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# 9. Calcular la precisión del modelo (Weierstrass)
def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

predictions_weierstrass = predict(X, theta_weierstrass)
accuracy_weierstrass = np.mean(predictions_weierstrass == y)
print(f"Precisión del modelo con el Método de Weierstrass: {accuracy_weierstrass * 100:.2f}%")

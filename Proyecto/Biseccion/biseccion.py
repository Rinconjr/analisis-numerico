import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Cargar el dataset con el separador correcto
df = pd.read_csv(r'../dataset_desercion_estudiantil.csv', sep=';')

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

# 3. Implementar el Método de la Bisección
def biseccion(X, y, iterations=10, tol=1e-5):
    m, n = X.shape
    # Inicializamos dos puntos iniciales para el método de la bisección
    theta_low = np.zeros((n, 1))  # Inicialización en cero
    theta_high = np.ones((n, 1)) * 10  # Inicialización en valores más altos
    
    for i in range(iterations):
        # Dividimos el intervalo a la mitad
        theta_mid = (theta_low + theta_high) / 2.0
        
        # Predicciones en los tres puntos
        h_low = sigmoid(X.dot(theta_low))
        h_mid = sigmoid(X.dot(theta_mid))
        h_high = sigmoid(X.dot(theta_high))
        
        # Gradientes en los tres puntos
        gradient_low = X.T.dot(h_low - y)
        gradient_mid = X.T.dot(h_mid - y)
        gradient_high = X.T.dot(h_high - y)
        
        # Decidir cuál mitad contiene la raíz, comparando signos de los gradientes
        if np.all(np.sign(gradient_mid) == np.sign(gradient_low)):
            theta_low = theta_mid
        else:
            theta_high = theta_mid
        
        # Verificar convergencia
        if np.linalg.norm(gradient_mid) < tol:
            print(f"Convergió en la iteración {i+1}")
            break
        
        # Mostrar el costo en cada iteración
        print(f"Iteración {i+1}: Costo = {log_likelihood(X, y, theta_mid)}")
    
    return theta_mid

# 5. Entrenar el modelo con el Método de la Bisección
theta_biseccion = biseccion(X, y, iterations=10)

# 6. Evaluar el modelo con probabilidades usando el método de la bisección
def predict_prob(X, theta):
    return sigmoid(X.dot(theta))  # Esto devuelve la probabilidad

# Predicciones con probabilidades usando el método de la bisección
probabilities_biseccion = predict_prob(X, theta_biseccion).flatten()
y = y.flatten()

# 7. Curva ROC para el método de la bisección
fpr, tpr, thresholds = roc_curve(y, probabilities_biseccion)
roc_auc = roc_auc_score(y, probabilities_biseccion)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.title('Curva ROC para Deserción de Estudiantes (Método de la Bisección)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')
plt.show()

# 8. Histograma de probabilidades para el método de la bisección
plt.figure(figsize=(10, 6))
plt.hist(probabilities_biseccion[y == 0], bins=20, alpha=0.5, label="No Deserta", color="blue")
plt.hist(probabilities_biseccion[y == 1], bins=20, alpha=0.5, label="Deserta", color="red")
plt.title("Distribución de Probabilidades de Deserción (Método de la Bisección)")
plt.xlabel("Probabilidad de Deserción")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# 9. Calcular la precisión del modelo (Bisección)
def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

predictions_biseccion = predict(X, theta_biseccion)
accuracy_biseccion = np.mean(predictions_biseccion == y)
print(f"Precisión del modelo con el Método de la Bisección: {accuracy_biseccion * 100:.2f}%")

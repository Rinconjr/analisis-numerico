#Descargar pip install pyarrow

import numpy as np
import pandas as pd

# 1. Cargar el dataset
# Cargamos el dataset desde el archivo CSV en la ruta proporcionada
df = pd.read_csv(r'dataset_desercion_estudiantil_promedio_ajustado_95.csv', sep=';')

# Preparamos los datos (features y etiquetas)
# Seleccionamos las características relevantes para la regresión logística
X = df[['Materias_inscritas', 'Promedio_estudiantil', 'Horas_estudio_por_semana', 'Edad']]  # Puedes ajustar las columnas
y = df['Desercion']

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
        # 4.1. Predicción
        h = sigmoid(X.dot(theta))
        
        # 4.2. Gradiente (Primera derivada)
        gradient = X.T.dot(h - y)
        
        # 4.3. Matriz Hessiana (Segunda derivada)
        H = (X.T.dot(np.diag((h * (1 - h)).reshape(-1))).dot(X))
        
        # 4.4. Actualización de los parámetros (Newton-Raphson)
        theta -= np.linalg.inv(H).dot(gradient)
        
        # Opcional: Mostrar el costo en cada iteración
        print(f"Iteración {i+1}: Costo = {log_likelihood(X, y, theta)}")
    
    return theta

# 5. Entrenar el modelo con Newton-Raphson
theta_final = newton_raphson(X, y, iterations=10)

# 6. Evaluar el modelo
def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

predictions = predict(X, theta_final)
accuracy = np.mean(predictions == y)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Opcional: Predicciones con el conjunto de prueba

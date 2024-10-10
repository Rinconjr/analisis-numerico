import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para la gráfica en 3D

# 1. Cargar el dataset con el separador correcto
df = pd.read_csv(r'../dataset_desercion_estudiantil.csv', sep=';')

# Preparamos los datos (solo Horas de estudio como característica)
X = df[['Horas_estudio_por_semana']]  # Solo la característica de Horas
y = df['Desercion']  # Etiqueta (0 o 1)

# Añadimos la columna de unos (intercepto) a X
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Convertimos y a un vector columna
y = y.values.reshape(-1, 1)

# Definir la función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Definir la función de costo (log-likelihood)
def log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1/m) * np.sum(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))

# Rango de valores para theta0 (intercepto) y theta1 (coeficiente de Horas_estudio_por_semana)
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-10, 10, 100)

# Crear una matriz para almacenar los valores de la función de costo
cost_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Calcular la función de costo para cada combinación de theta0 y theta1
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta = np.array([theta0_vals[i], theta1_vals[j]]).reshape(-1, 1)
        cost_vals[i, j] = log_likelihood(X, y, theta)

# Graficar la función de costo en 3D
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, cost_vals, cmap='viridis')

# Añadir etiquetas
ax.set_title('Superficie de la Función de Costo')
ax.set_xlabel('Theta 0 (Intercepto)')
ax.set_ylabel('Theta 1 (Horas de Estudio)')
ax.set_zlabel('Costo (Log-Likelihood)')
plt.show()

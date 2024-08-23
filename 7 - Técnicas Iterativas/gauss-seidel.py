# Tecnica iterativa gauss-seidel para resolver sistemas de ecuaciones lineales
import numpy as np

def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.copy(x0)

    print(f"Iteración 0: x1 = {x[0]:.5f}, x2 = {x[1]:.5f}, x3 = {x[2]:.5f}")

    for iteration in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        print(f"Iteración {iteration + 1}: {[f'{val:.5f}' for val in x_new]}")  # Mostrar con 5 decimales
        if np.linalg.norm(x_new - x) < tol:
            print(f"Convergió en la iteración {iteration + 1}")
            return x_new
        x = x_new
    return x

# Ejemplo de uso
# Sistema de ecuaciones:
A = np.array([[1, 2, -2],
              [1, 1, 1],
              [2, 2, 1]])

# Vector solución:
b = np.array([7, 2, 5])

# Vector de valores iniciales:
x0 = np.zeros_like(b)

# Tolerancia:
tol = 1e-10

# Número máximo de iteraciones:
max_iter = 1000

# Llamado a la función:
x = gauss_seidel(A, b, x0, tol, max_iter)

print('Solución final:', [f'{val:.5f}' for val in x])
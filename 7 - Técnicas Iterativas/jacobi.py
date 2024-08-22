# Tecnica iterativa jacobi para resolver sistemas de ecuaciones lineales

import numpy as np

def jacobi(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.copy(x0)
    for iteration in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        print(f"Iteración {iteration + 1}: {[f'{val:.5f}' for val in x_new]}")  # Mostrar con 5 decimales
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Ejemplo de uso
A = np.array([[5, -2, 1],
              [2, 5, -2],
              [1, -4, 3]])

b = np.array([24, -14, 26])

x0 = np.zeros_like(b)

tol = 1e-10

max_iter = 1000

x = jacobi(A, b, x0, tol, max_iter)

print('Solución final:', [f'{val:.5f}' for val in x])

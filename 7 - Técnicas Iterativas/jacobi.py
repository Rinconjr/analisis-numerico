import numpy as np

def jacobi(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.copy(x0)

    print(f"Iteración 0: x1 = {x[0]:.5f}, x2 = {x[1]:.5f}, x3 = {x[2]:.5f}")

    for iteration in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        print(f"Iteración {iteration + 1}: x1 = {x_new[0]:.5f}, x2 = {x_new[1]:.5f}, x3 = {x_new[2]:.5f}")
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Convergió en la iteración {iteration + 1}")
            return x_new
        x = np.copy(x_new)  # Actualizar x para la siguiente iteración
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
x = jacobi(A, b, x0, tol, max_iter)

print('Solución final:', f"x1 = {x[0]:.5f}, x2 = {x[1]:.5f}, x3 = {x[2]:.5f}")

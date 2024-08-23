"""
LU - GRUPO 1:
- Nicolás Montañez
- Juan Esteban Muñoz
- Johan Espitia
- Nicolás Montaño
"""

import numpy as np

def lu_factorization(A):
    """
    Factoriza la matriz A en el producto de una matriz triangular inferior L y una matriz triangular superior U.

    Parámetros:
    A (numpy.ndarray): Matriz cuadrada de entrada para la factorización LU.

    Retorna:
    tuple: Matrices L y U tales que A = L * U, donde L es triangular inferior con unos en la diagonal y U es triangular superior.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Inicializar la diagonal de L en 1
    for i in range(n):
        L[i, i] = 1

    # Cálculo de las matrices L y U
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            if U[i, i] == 0:
                L[j, i] = 0
            else:
                L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U

# Ejemplo de uso
A = np.array([[2,-1, 1],
              [3, 3, 9],
              [3, 3, 5]])

L, U = lu_factorization(A)
print("Matriz L:")
print(L)
print("Matriz U:")
print(U)
print("Producto LU:")
print(L @ U)
"""
PLU - GRUPO 1:
- Johan Espitia
- Nicolás Montañez
- Nicolás Montaño
- Juan Esteban Muñoz
"""

import numpy as np


def plu_decomposition(a, p_transposed=False):
    n = len(a)
    p = np.eye(n)
    l = np.zeros((n, n))
    u = a.copy()

    for i in range(n):
        # Paso 1: Realizar la permutación si es necesario
        max_index = np.argmax(abs(u[i:, i])) + i
        if i != max_index:
            u[[i, max_index]] = u[[max_index, i]]
            p[[i, max_index]] = p[[max_index, i]]
            if i > 0:
                l[[i, max_index], :i] = l[[max_index, i], :i]

        # Paso 2: Realizar la eliminación para formar L y U
        for j in range(i + 1, n):
            factor = u[j, i] / u[i, i]
            l[j, i] = factor
            u[j, i:] -= factor * u[i, i:]

    # Colocar unos en la diagonal de L
    np.fill_diagonal(l, 1)

    if not p_transposed:
        return p, l, u
    else:
        return p.T, l, u


def tranpose(a):
    for i in range(len(a)):
        for j in range(i, len(a)):
            a[i][j], a[j][i] = a[j][i], a[i][j]
    return a


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length.")
    return sum(ai * bi for ai, bi in zip(a, b))


def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]

    return y


def backward_substitution(U, b):
    n = len(b)
    x = np.zeros(n)

    # Perform backward substitution
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Matrix is singular.")
        x[i] = (b[i] - dot_product(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def solve_plu(P, L, U, b):
    P = tranpose(P)
    Pb = dot_product(P, b)
    y = forward_substitution(L, Pb)
    x = backward_substitution(U, y)

    return x

# Ejemplo de uso
A = np.array([[2, -1, 1],
              [3, 3, 9],
              [3, 3, 5]])

P, L, U = plu_decomposition(A)
b = np.array([2, 1, 3])

x = solve_plu(P, L, U, b)
print("Solución x:")
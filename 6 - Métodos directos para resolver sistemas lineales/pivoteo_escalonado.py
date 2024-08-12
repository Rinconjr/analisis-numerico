import numpy as np

def pivoteo(A, b, k):
    n = len(A)
    e = np.zeros(n)

    for i in range(k, n):
        e[i] = abs(A[i, k]) / abs(max(A[i, :]))

    f = np.argmax(e)
    A[[k, f], :] = A[[f, k], :]
    b[[k, f]] = b[[f, k]]

    return A, b

def elim_gauss(A, b, piv):
    n = len(A)

    # Eliminación hacia adelante
    for k in range(n - 1):
        if piv != 0:
            A, b = pivoteo(A, b, k)
        for i in range(k + 1, n):
            F = A[i, k] / A[k, k]
            A[i, :] = A[i, :] - F * A[k, :]
            b[i] = b[i] - F * b[k]

    return A, b

def back_substitution(A, b):
    n = len(A)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

# Definición del sistema de ecuaciones
A = np.array([[2.0, 3.0, 0.0], [-1.0, 2.0, -1.0], [3.0, 0.0, 2.0]])
b = np.array([8.0, 0.0, 9.0])

# Proceso de eliminación gaussiana con pivoteo
piv = 1  # Activar pivoteo
A_final, b_final = elim_gauss(A, b, piv)

# Proceso de sustitución hacia atrás para hallar la solución
x = back_substitution(A_final, b_final)

# Resultados
print("Matriz A después de la eliminación:\n", A_final)
print("Vector b después de la eliminación:\n", b_final)
print("La solución del sistema es:\n", x)

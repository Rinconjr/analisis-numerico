import numpy as np

def factorizacion_LU(n, A, L, U):
  # Paso 1: Seleccionar l11 y u11 al satisfacer l11*u11=a11
  # si a11 = 0, se debe permutar la fila 1 con otra fila
    if A[0][0] == 0:
        print('Factorizacion imposible, se debe permutar...')
        permutacion(A)
        if A is None:
            print('No se puede factorizar')
            return None, None
    else:
        # Calcular L y U
        for i in range(n):
            for j in range(i, n):
                U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            for j in range(i + 1, n):
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
        return L, U



def permutacion(A):
  pass

def imprimir_matriz(matrix):
    for row in matrix:
        print(row)

# Definir las entradas de la funcion
# Matriz a factorizar
A = np.array([[2, -1, 1],
             [3, 3, 9],
             [3, 3, 5]])
# Tamaño de la matriz
n = len(A)
# Matrices L y U (Identidad)
L = np.eye(3)
U = np.eye(3)

# Llamar a la función factorizacion_LU
L, U = factorizacion_LU(n, A, L, U)

# Mostrar resultados de la factorizacion
print('A:')
imprimir_matriz(A)
print('L:')
imprimir_matriz(L)
print('U:')
imprimir_matriz(U)











print("-------------------")
# Dividir la primera fila por 2 y reemplazarla
# 1/2E1 + E2 -> E2
# A[1]=A[0]/2 + A[1]
# imprimir_matriz(A)

# Llamar a la función factorizacion_LU
# factorizacion_LU(A)

# Mostrar resultados de la factorizacion


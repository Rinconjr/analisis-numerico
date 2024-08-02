def factorizacion_LU(n, A, L, U):
    L = [[0 for x in range(n)] for y in range(n)]
    U = [[0 for x in range(n)] for y in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i][j] = 1
            if i > j:
                factor = A[j][i] / A[i][i]
                L[j][i] = factor
                for k in range(i, n):
                    A[j][k] = A[j][k] - factor * A[i][k]
            if i < j:
                U[i][j] = A[i][j]
    return L, U


def factorizacion(n, A, L, U):
  # Revisar si a_1_1 es distinto de 0
  if A[0][0] == 0:
    print('Factorizacion imposible, se debe permutar...')


def permutacion():
  pass

# Función para imprimir una matriz
def imprimir_matriz(matrix):
    for row in matrix:
        print(row)

# Definir las entradas de la funcion
n = 3
A = [[2, -1, 1],
     [3, 3, 9],
     [3, 3, 5]]

L = [[1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]

U = [[1, 0, 0],
  [0, 1, 0],
  [0, 0, 1]]


# Llamar a la función factorizacion_LU
L, U = factorizacion_LU(n, A, L, U)

# Mostrar resultados de la factorizacion
print("Matriz L")
imprimir_matriz(L)
print("\n")
print("Matriz U")
imprimir_matriz(U)

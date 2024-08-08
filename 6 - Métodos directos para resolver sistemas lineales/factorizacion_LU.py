import numpy as np

def factorizacion_LU(n, A):
    # Inicializar L y U
    L = np.eye(n)
    U = np.zeros((n, n))  # Cambiar a ceros para U

    # Paso 1: Seleccionar l11 y u11 al satisfacer l11 * u11 = a11
    if A[0][0] == 0:
        print('Factorización imposible, se debe permutar...')
        A = permutacion(A)
        if A is None:
            print('No se puede factorizar')
            return None, None
    
    # Paso 2: Calcular la primera fila de U y la primera columna de L
    for j in range(n):
        U[0][j] = A[0][j]  # Asignar directamente de A a U
    for j in range(1, n):
        L[j][0] = A[j][0] / A[0][0]  # Primera columna de L

    # Paso 3: Para i = 1, ..., n-1, hacer los pasos 4 y 5.
    for i in range(1, n):
        # Paso 4: Seleccionar lii y uii al satisfacer lii * uii = aii - sum(lik * uki)
        U[i][i] = A[i][i] - sum(L[i][k] * U[k][i] for k in range(i))
        L[i][i] = 1  # Los elementos en la diagonal de L son 1

        if U[i][i] == 0:
            print('Factorización imposible, se debe permutar...')
            A = permutacion(A)
            if A is None:
                print('No se puede factorizar')
                return None, None
            return factorizacion_LU(n, A)

        # Paso 5: Calcular la i-ésima fila de U y la i-ésima columna de L
        for j in range(i + 1, n):
            U[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(i)))
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    # Paso 6: Seleccionar lnn y unn al satisfacer lnn * unn = ann - sum(lnk * ukn)
    U[n-1][n-1] = A[n-1][n-1] - sum(L[n-1][k] * U[k][n-1] for k in range(n-1))
    L[n-1][n-1] = 1  # Los elementos en la diagonal de L son 1

    if L[n-1][n-1] * U[n-1][n-1] == 0:
        print('Factorización imposible, A es singular.')

    # Paso 7: Salida de matrices L y U
    return L, U

# Se recorre la matriz A para permutar filas
def permutacion(A):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if A[j][i] != 0:
                imprimir_matriz(A)
                A[[i, j]] = A[[j, i]]  # Intercambiar filas
                print('Permutación de filas...')
                imprimir_matriz(A)
                return A
    return None  # No se puede permutar

def imprimir_matriz(matrix):
    for row in matrix:
        print(row)

# Definir las entradas de la función
A = np.array([[2,-1, 1],
              [3, 3, 9],
              [3, 3, 5]])
n = len(A)

B = A.copy()
# Llamar a la función factorizacion_LU
L, U = factorizacion_LU(n, B)

# Mostrar resultados de la factorización
print('A:')
imprimir_matriz(A)
print('L:')
imprimir_matriz(L)
print('U:')
imprimir_matriz(U)

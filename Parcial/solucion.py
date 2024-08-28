# Esta es la solución general que muestra las primeras 100 iteraciones de Gauss-Seidel y Jacobi para n=6 a n=15.
# Se usa para ver la convergencia de los métodos y comparar los resultados obtenidos.

import numpy as np

def generar_matriz_viga(n):
    A = np.zeros((n, n))

    for i in range(n):
        if i == 0:
            A[i, i] = 12
            A[i, i+1] = -6
            A[i, i+2] = 4/3
        elif i == 1:
            A[i, i-1] = -4
            A[i, i] = 6
            A[i, i+1] = -4
            A[i, i+2] = 1
        elif i == n-1:
            A[i, i-2] = 4/3
            A[i, i-1] = 6
            A[i, i] = -12
        elif i == n-2:
            A[i, i-2] = 1
            A[i, i-1] = -4
            A[i, i] = 6
            A[i, i+1] = -4
        else:
            A[i, i-2] = 1
            A[i, i-1] = -4
            A[i, i] = 6
            A[i, i+1] = -4
            A[i, i+2] = 1
    return A
    

def generar_matriz_viga_voladizo(n):
    # Crear una matriz de ceros de tamaño n x n
    A = np.zeros((n, n))

    # Llenar la matriz
    for i in range(n):
        if i == 0:
            A[i, i] = 12
            if i + 1 < n:
                A[i, i + 1] = -6
            if i + 2 < n:
                A[i, i + 2] = 4/3
        elif i == n - 2:  # Penúltima fila
            A[i, i-2] = 1
            A[i, i - 1] = -93/25
            A[i, i] = 111/25
            A[i, i + 1] = -43/25
        elif i == n - 1:  # Última fila
            A[i, i - 2] = 12/25
            A[i, i - 1] = 24/25
            A[i, i] = 12/25
        else:
            A[i, i-2] = 1
            A[i, i - 1] = -4
            A[i, i] = 6
            A[i, i + 1] = -4
            if i + 2 < n:
                A[i, i + 2] = 1

    return A

def imprimir_matriz(A):
    n = A.shape[0]
    encabezado = "    " + " ".join(["{:>6}".format(j + 1) for j in range(n)])
    print(encabezado)
    print("   " + "-" * (7 * n))

    for i in range(n):
        fila = ["{:6.3f}".format(A[i, j]) for j in range(n)]
        print("{:>2} | {}".format(i + 1, " ".join(fila)))

def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.copy(x0)

    # Mostrar los valores de x en cada iteración
    for iteration in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        # Mostrar los valores de x en cada iteración
        print(f"Iteración {iteration + 1}: {[f'{val:.5f}' for val in x_new]}")

        if np.linalg.norm(x_new - x) < tol:
            print(f"Convergió en la iteración {iteration + 1}")
            return x_new
        
        x = x_new
    return x

def jacobi(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.copy(x0)

    # Mostrar los valores de x en cada iteración
    for iteration in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        # Mostrar los valores de x en cada iteración
        print(f"Iteración {iteration + 1}: {[f'{val:.5f}' for val in x_new]}")

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Convergió en la iteración {iteration + 1}")
            return x_new
        
        x = np.copy(x_new)  # Actualizar x para la siguiente iteración
    return x

# Función para generar el vector b con una "flecha" en el centro
def generar_vector_b(n):
    b = np.zeros(n)
    centro = n // 2
    valor_minimo = 10  # Valor mínimo en los extremos
    valor_maximo = 100  # Valor máximo en el centro

    # Calcular pendiente de la disminución lineal
    pendiente = (valor_maximo - valor_minimo) / (centro if centro != 0 else 1)

    # Asignar valores decrecientes desde el centro hacia los extremos
    for i in range(n):
        distancia_al_centro = abs(i - centro)
        b[i] = valor_maximo - (distancia_al_centro * pendiente)
        # Asegurar que no sea menor que el valor mínimo
        b[i] = max(b[i], valor_minimo)

    return b

# Calcular error relativo entre la solución actual y la solución de referencia
def calcular_error(x_sol, x_ref):
    return np.linalg.norm(x_sol - x_ref) / np.linalg.norm(x_ref)

# Realizar el proceso iterativo para varios tamaños de n
def resolver_sistema_gauss_jacobi():
    tol = 1e-5
    max_iter = 100

    # IMPORTANTE: Modificar aqui para analisis de resultados
    for n in range(6, 8):  # De n=6 a n=8
        print(f"\nResolviendo sistema con n = {n}")

        # Generar matriz A y vector b
        A = generar_matriz_viga(n)
        b = generar_vector_b(n)
        x0 = np.zeros(n)  # Aproximación inicial (0, 0, ..., 0)

        print("Matriz A:")
        imprimir_matriz(A)
        print("\nVector b:", b, "\n")

        # Calcular número de condición de A
        cond_A = np.linalg.cond(A)
        print(f"Número de condición de la matriz A: {cond_A:.5e}")

        # Resolver usando Gauss-Seidel
        print("\nSolución con Gauss-Seidel:")
        gauss_seidel(A, b, x0, tol, max_iter)
        
        # Resolver usando Jacobi
        print("\nSolución con Jacobi:")
        jacobi(A, b, x0, tol, max_iter)

# Ejecutar el proceso para varios tamaños de n
resolver_sistema_gauss_jacobi()
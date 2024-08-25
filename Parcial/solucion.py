import numpy as np

def generar_matriz_viga(n):
    A = np.zeros((n, n)) # Crear una matriz de ceros de tamaño n x n

    # Asignar los valores de la diagonal principal
    for i in range(n):
        if i == 0:
            A[i, i] = 12
        elif i == n - 1:
            A[i, i] = -12
        else:
            A[i, i] = 6

    # Asignar los valores de las diagonales superiores e inferiores
    for i in range(n - 1):
        if i == 0 or i == n - 2:
            A[i, i + 1] = -6
            A[i + 1, i] = -6
        else:
            A[i, i + 1] = -4
            A[i + 1, i] = -4

    # Asignar los valores de las segundas diagonales superiores e inferiores
    for i in range(n - 2):
        if i == 0 or i == n - 3:
            A[i, i + 2] = 4 / 3
            A[i + 2, i] = 4 / 3
        else:
            A[i, i + 2] = 1
            A[i + 2, i] = 1

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
    print(f"Iteración 0: x1 = {x[0]:.5f}, x2 = {x[1]:.5f}, x3 = {x[2]:.5f}")

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
    print(f"Iteración 0: x1 = {x[0]:.5f}, x2 = {x[1]:.5f}, x3 = {x[2]:.5f}")

    for iteration in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        # Mostrar los valores de x en cada iteración
        print(f"Iteración {iteration + 1}: x1 = {x_new[0]:.5f}, x2 = {x_new[1]:.5f}, x3 = {x_new[2]:.5f}")

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

    for n in range(6, 16):  # De n=6 a n=8
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
        x_gs = gauss_seidel(A, b, x0, tol, max_iter)
        
        # Resolver usando Jacobi
        print("\nSolución con Jacobi:")
        x_jacobi = jacobi(A, b, x0, tol, max_iter)

        # Comparar errores relativos entre ambos métodos (usando Gauss-Seidel como referencia)
        error_jacobi_vs_gs = calcular_error(x_jacobi, x_gs)
        print(f"\nError relativo Jacobi vs Gauss-Seidel: {error_jacobi_vs_gs:.5e}")

# Ejecutar el proceso para varios tamaños de n
resolver_sistema_gauss_jacobi()

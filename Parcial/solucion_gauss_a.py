# Solucion con matriz del punto a, se calculan las tres primeras iteraciones con aritmética de corte y redondeo con sus respectivos
# errores relativos y absolutos. Se muestra el número de condición de la matriz A. Se resuelve el sistema para n=6 a n=15.

import numpy as np

def generar_matriz_viga(n):
    A = np.zeros((n, n))  # Crear una matriz de ceros de tamaño n x n

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

def gauss_seidel_corte(A, b, x0):
    n = len(A)
    x = np.copy(x0)

    print(f"\n--- Primeras 3 Iteraciones con Aritmética de Corte ---")
    for iteration in range(3):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))

            # Aplicar corte a tres dígitos en las operaciones
            s1 = int(s1 * 1000) / 1000
            s2 = int(s2 * 1000) / 1000
            x_new[i] = int((b[i] - s1 - s2) / A[i][i] * 1000) / 1000

        # Cálculo de errores relativo y absoluto
        error_relativo = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
        error_absoluto = np.linalg.norm(x_new - x)

        # Mostrar detalles de la iteración solo una vez por iteración
        print(f"Iteración {iteration + 1}: {[f'{val:.3f}' for val in x_new]}, "
              f"Error Relativo: {error_relativo:.3e}, Error Absoluto: {error_absoluto:.3e}")

        x = x_new  # Actualizar x para la siguiente iteración


def gauss_seidel_redondeo(A, b, x0):
    n = len(A)
    x = np.copy(x0)

    print(f"\n--- Primeras 3 Iteraciones con Redondeo ---")
    for iteration in range(3):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))

            # Aplicar redondeo a tres dígitos en las operaciones
            s1 = round(s1, 3)
            s2 = round(s2, 3)
            x_new[i] = round((b[i] - s1 - s2) / A[i][i], 3)

        # Cálculo de errores relativo y absoluto
        error_relativo = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
        error_absoluto = np.linalg.norm(x_new - x)

        # Mostrar detalles de la iteración
        print(f"Iteración {iteration + 1}: {[f'{val:.3f}' for val in x_new]}, "
              f"Error Relativo: {error_relativo:.3e}, Error Absoluto: {error_absoluto:.3e}")

        x = x_new

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

# Realizar el proceso iterativo para varios tamaños de n
def resolver_sistema_viga_articulado():
    for n in range(6, 16):  # De n=6 a n=15
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

        # Resolver usando Gauss-Seidel con aritmética de corte
        gauss_seidel_corte(A, b, x0)

        # Resolver usando Gauss-Seidel con redondeo
        gauss_seidel_redondeo(A, b, x0)

# Ejecutar el proceso para varios tamaños de n
resolver_sistema_viga_articulado()


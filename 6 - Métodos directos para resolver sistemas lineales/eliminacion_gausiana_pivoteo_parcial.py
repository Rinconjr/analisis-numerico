import numpy as np

def pivoteo_parcial(matrizA, vectorb):
    A = matrizA
    b = vectorb
    m, n = A.shape
    pos = 0
    # Se halla la matriz aumentada
    G = np.hstack((A, b.reshape(-1, 1)))

    if m == n:
        if np.linalg.det(A) != 0:
            for i in range(m):  # El primer contador recorre las filas
                for l in range(i, m):
                    mayor = abs(G[i, i])
                    c = abs(G[l, i])
                    if c >= mayor:
                        mayor = c
                        pos = l
                    if c < mayor:
                        pos = i
                Vector = G[i, :].copy()
                G[i, :] = G[pos, :]
                G[pos, :] = Vector
                for k in range(i + 1, m):
                    aux = -G[k, i] / G[i, i]  # Número que convierte en ceros las posiciones debajo del pivote
                    for j in range(n + 1):  # El tercer contador recorre las columnas
                        G[k, j] = G[k, j] + (aux * G[i, j])  # i es constante mientras k y j recorren todas las posiciones

            X = np.zeros(m)
            for h in range(n - 1, -1, -1):  # Se hace una sustitución regresiva
                suma = 0  # Esta suma es un elemento de ayuda para el despeje
                for p in range(n):
                    suma = suma + G[h, p] * X[p]
                X[h] = (G[h, m] - suma) / G[h, h]  # Despeje de la variable en la posición (h,1)

            print('')
            print('MATRIZ AUMENTADA G')
            print(G)
            print('')
            print('VECTOR X ')
            print(X)
            print('')
        else:
            print('El sistema no es cuadrado y no tiene única solución')
    else:
        print('El sistema no es cuadrado y no tiene única solución')

# Ejemplo de uso:
matrizA = np.array([[8, 2, -2], 
                    [10, 2, 4], 
                    [12, 2, 2]], dtype=float)
vectorb = np.array([-2, 4, 6], dtype=float)

pivoteo_parcial(matrizA, vectorb)

import numpy as np

def sustitucionHaciaAtras(A, b):
    n = len(b)
    # Construir la matriz aumentada
    matriz_aum = np.hstack((A, b.reshape(-1, 1)))
    # Proceso de eliminación
    for i in range(n):
        # Encontrar el pivote
        p = i
        while p < n and matriz_aum[p, i] == 0:
            p += 1
        if p == n:
            return "No existe una solución única"
        
        # Intercambiar filas si es necesario
        if p != i:
            matriz_aum[[i, p]] = matriz_aum[[p, i]]
        
        # Eliminar las variables de las filas inferiores
        for j in range(i + 1, n):
            m_ji = matriz_aum[j, i] / matriz_aum[i, i]
            matriz_aum[j] = matriz_aum[j] - m_ji * matriz_aum[i]
    # Verificar si no existe una solución única
    if matriz_aum[n-1, n-1] == 0:
        return "No existe una solución única"
    
    # Sustitución hacia atrás
    x = np.zeros(n)
    x[n-1] = matriz_aum[n-1, n] / matriz_aum[n-1, n-1]
    print(x)
    for i in range(n-2, -1, -1):
        sum_ax = sum(matriz_aum[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (matriz_aum[i, n] - sum_ax) / matriz_aum[i, i]
    
    return x

# Ejemplo de uso:
A = np.array([[3, 2, -2],
              [2, -1, 3],
              [1, 4, 2]], dtype=float)

b = np.array([0, 9, -4], dtype=float)

solution = sustitucionHaciaAtras(A, b)
print("Solución:", solution)

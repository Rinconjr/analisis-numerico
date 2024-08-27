import numpy as np

def cholesky_decomposition(A):
    """
    Realiza la factorización de Cholesky de una matriz A.
    
    Args:
    A (numpy.ndarray): Matriz simétrica y definida positiva.
    
    Returns:
    L (numpy.ndarray): Matriz triangular inferior tal que A = L * L.T
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            
            if i == j:  # Diagonal elements
                L[i][j] = np.sqrt(A[i][i] - sum_k)
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]
    
    return L

# Ejemplo
A = np.array([[25, 15, -5],
              [15, 18, 0],
              [-5, 0, 11]])

# Verificamos que A es simétrica y definida positiva
try:
    L = cholesky_decomposition(A)
    print("Matriz A:")
    print(A)
    print("\nMatriz L (resultado de la factorización de Cholesky):")
    print(L)
    print("\nVerificación (L * L.T):")
    print(np.dot(L, L.T))
except np.linalg.LinAlgError as e:
    print(f"Error en la factorización de Cholesky: {e}")

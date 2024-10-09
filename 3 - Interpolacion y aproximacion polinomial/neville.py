import numpy as np

def neville(x_data, y_data, x):
    n = len(x_data)
    Q = np.zeros((n, n))  # Creamos una tabla vacía para Q

    # Inicializamos la primera columna de Q con los valores de y_data (f(xi))
    for i in range(n):
        Q[i, 0] = y_data[i]

    # Implementamos la fórmula recursiva de Neville
    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i, j] = ((x - x_data[i - j]) * Q[i, j - 1] - (x - x_data[i]) * Q[i - 1, j - 1]) / (x_data[i] - x_data[i - j])

    return Q[n - 1, n - 1], Q  # El valor interpolado es Q[n-1, n-1]

# Valores de la tabla 3.5
x_data = [1.0, 1.3, 1.6, 1.9, 2.2]  # Puntos x0, x1, x2, x3, x4
y_data = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]  # Valores f(x0), f(x1), ..., f(x4)

# Punto donde queremos evaluar el polinomio
x = 1.5  

result, table = neville(x_data, y_data, x)
print(f"El valor interpolado en x = {x} es: {result}")
print("Tabla de Neville:")
print(table)

import math

# Definir la función f(x) = x * e^x - 10
def f(x):
    return x * math.exp(x) - 10

# Derivada de f(x), f'(x) = e^x * (1 + x)
def f_prime(x):
    return math.exp(x) * (1 + x)

# Método de Newton-Raphson con impresión de iteraciones
def newton_raphson(f, f_prime, x0, tol=1e-5, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_next = x - f(x) / f_prime(x)
        print(f"Iteración {i+1}: x = {x_next}")  # Imprimir el valor de x en cada iteración
        if abs(x_next - x) < tol:
            return x_next, i + 1  # Solución encontrada y número de iteraciones
        x = x_next
    return None, max_iter  # Si no converge dentro del número máximo de iteraciones

# Parámetros iniciales
p0 = 1.5  # Aproximación inicial
TOL = 1e-5  # Tolerancia
N0 = 100  # Número máximo de iteraciones

# Aplicar el método de Newton-Raphson
solucion_newton, iteraciones_newton = newton_raphson(f, f_prime, p0, TOL, N0)

# Mostrar los resultados
if solucion_newton is not None:
    print(f"\nSolución aproximada con Newton-Raphson: {solucion_newton} en {iteraciones_newton} iteraciones")
else:
    print(f"El método de Newton-Raphson falló después de {N0} iteraciones.")

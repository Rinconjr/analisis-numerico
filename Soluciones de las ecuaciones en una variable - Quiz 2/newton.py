def newton(f, f_prime, p0, TOL, N0):
    i = 1
    
    while i <= N0:
        # Paso 3: Calcular la nueva aproximación
        p = p0 - f(p0) / f_prime(p0)
        
        # Paso 4: Verificar si la aproximación es suficiente
        if abs(p - p0) < TOL:
            return p, i  # Procedimiento exitoso, retornamos la raíz y el número de iteraciones
        
        # Paso 5: Incrementar el contador de iteraciones
        i += 1
        
        # Paso 6: Actualizar p0 con la nueva aproximación
        p0 = p
    
    # Paso 7: Si no converge dentro de N0 iteraciones
    print(f"El método falló después de {N0} iteraciones.")
    return None, i

# Ejemplo de uso:
# Definimos la función f(x) = x^3 + 4x^2 - 10 y su derivada f'(x)
def f(x):
    return x**3 + 4*x**2 - 10

def f_prime(x):
    return 3*x**2 + 8*x  # Derivada de f(x)

# Parámetros iniciales
p0 = 1.5  # Aproximación inicial
TOL = 1e-4  # Tolerancia
N0 = 100  # Número máximo de iteraciones

# Ejecutamos el método de Newton
raiz, iteraciones = newton(f, f_prime, p0, TOL, N0)

if raiz is not None:
    print(f"La solución aproximada es: {raiz} en {iteraciones} iteraciones")
else:
    print(f"El método falló después de {iteraciones} iteraciones.")

# Metodo mas eficiente que el de Newton porque
# El metodo de la secante no necesita calcular la derivada, en su lugar,
# utiliza una proximacion de la derivada basada en dos puntos cercanos sobre la curva de la funcion.

def secante(f, p0, p1, TOL, N0):
    q0 = f(p0)
    q1 = f(p1)
    i = 2  # Ya usamos dos puntos iniciales

    while i <= N0:
        # Paso 3: Calcular la nueva aproximación
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        
        # Paso 4: Verificar si la aproximación es suficiente
        if abs(p - p1) < TOL:
            return p, i  # Procedimiento exitoso
        
        # Paso 5: Incrementar el contador de iteraciones
        i += 1
        
        # Paso 6: Actualizar p0, p1, q0, q1 para la siguiente iteración
        p0 = p1
        q0 = q1
        p1 = p
        q1 = f(p1)
    
    # Paso 7: Si no converge dentro de N0 iteraciones
    print(f"El método falló después de {N0} iteraciones.")
    return None, i

# Ejemplo de uso:
# Definimos la función f(x) = x^3 + 4x^2 - 10
def f(x):
    return x**3 + 4*x**2 - 10

# Parámetros iniciales
p0 = 1  # Primer punto inicial
p1 = 2  # Segundo punto inicial
TOL = 1e-4  # Tolerancia
N0 = 100  # Número máximo de iteraciones

# Ejecutamos el método de la secante
raiz, iteraciones = secante(f, p0, p1, TOL, N0)

if raiz is not None:
    print(f"La solución aproximada es: {raiz} en {iteraciones} iteraciones")
else:
    print(f"El método falló después de {iteraciones} iteraciones.")

import math

def punto_fijo(g, p0, TOL, N0):
    # Paso 1
    i = 1
    
    # Paso 2
    while i <= N0:
        # Paso 3
        p = g(p0)
        
        # Mostrar la solución actual en cada iteración
        print(f"Iteración {i}: p = {p}")
        
        # Paso 4
        if abs(p - p0) < TOL:
            return p, i  # Procedimiento completado exitosamente, retorna también las iteraciones
        
        # Paso 5
        i += 1
        
        # Paso 6
        p0 = p  # Actualizar p0
    
    # Paso 7
    print(f"El método falló después de {N0} iteraciones.")
    return None, i

# Definimos la función g(x) como g2(x) = ln(10 / x)
def g(x):
    return math.log(10 / x)

# Parámetros iniciales
p0 = 1.5  # Aproximación inicial
TOL = 1e-4  # Tolerancia
N0 = 100  # Número máximo de iteraciones

# Ejecutamos el método del punto fijo
raiz, iteraciones = punto_fijo(g, p0, TOL, N0)

if raiz is not None:
    print(f"\nLa solución aproximada es: {raiz} en {iteraciones} iteraciones")
else:
    print(f"El método falló después de {iteraciones} iteraciones.")

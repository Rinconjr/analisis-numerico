import math

def newton(f, f_prime, p0, TOL, N0):
    i = 1

    #print(f"Iteración {i}: p = {p0}")
    
    while i <= N0:
        # Paso 3: Calcular la nueva aproximación
        try:
            p = p0 - f(p0) / f_prime(p0)
        except ZeroDivisionError:
            print(f"Error en la iteración {i}: división por cero.")
            return None, i
        
        # Paso 4: Verificar si la aproximación es suficiente
        if abs(p - p0) < TOL:
            return p, i  # Procedimiento exitoso, retornamos la raíz y el número de iteraciones
        
        # Paso 5: Incrementar el contador de iteraciones
        i += 1

        #print(f"Iteración {i}: p = {p}")
        
        # Paso 6: Actualizar p0 con la nueva aproximación
        p0 = p
    
    # Paso 7: Si no converge dentro de N0 iteraciones
    #print(f"El método falló después de {N0} iteraciones.")
    return None, i

# Definimos la función f(x) = x * e^x - 10 y su derivada f'(x) = e^x + x * e^x
def f(x):
    return x * math.exp(x) - 10

def f_prime(x):
    return math.exp(x) + x * math.exp(x)  # Derivada de f(x)

# Parámetros iniciales
p0 = 2  # Aproximación inicial cercana a la solución
TOL = 1e-6  # Tolerancia
N0 = 100  # Número máximo de iteraciones

# Ejecutamos el método de Newton
raiz, iteraciones = newton(f, f_prime, p0, TOL, N0)

if raiz is not None:
    print(f"La solución aproximada es: {raiz} en {iteraciones} iteraciones")
#else:
    #print(f"El método falló después de {iteraciones} iteraciones.")

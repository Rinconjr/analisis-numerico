import math

def punto_fijo(g, p0, TOL, N0):
    # Paso 1: Inicializamos la iteración
    i = 1
    
    # Paso 2: Iteramos hasta que se alcance la tolerancia o el número máximo de iteraciones
    while i <= N0:
        # Paso 3: Calculamos la siguiente aproximación
        try:
            p = g(p0)
        except ValueError as e:
            # Si ocurre un error matemático (como logaritmo de un valor negativo), mostramos el error
            print(f"Error en la iteración {i}: {e}")
            return None, i
        
        # Paso 4: Verificamos si la aproximación es suficientemente precisa
        if abs(p - p0) < TOL:
            return p, i  # Si cumple la tolerancia, retornamos la solución y las iteraciones
        
        # Paso 5: Incrementamos el contador de iteraciones
        i += 1
        
        # Paso 6: Actualizamos p0 para la siguiente iteración
        p0 = p
    
    # Paso 7: Si no convergió dentro de las iteraciones permitidas, mostramos un mensaje de fallo
    #print(f"El método falló después de {N0} iteraciones.")
    return None, i

# Definimos las dos funciones g1 y g2 para la ecuación xe^x = 10
def g1(x):
    # g1(x) = 10 / e^x
    return 10 / math.exp(x)

def g2(x):
    # g2(x) = ln(10 / x) - Agregamos validación para evitar logaritmo de valores no válidos
    if x <= 0:
        raise ValueError("Logaritmo indefinido para x <= 0")
    return math.log(10 / x)

# Parámetros iniciales
p0 = 2  # Aproximación inicial, valor cercano a la solución esperada
TOL = 1e-6  # Tolerancia
N0 = 100  # Número máximo de iteraciones

# Ejecutamos el método del punto fijo para g1
raiz1, iteraciones1 = punto_fijo(g1, p0, TOL, N0)
if raiz1 is not None:
    print(f"Solución con g1: {raiz1} en {iteraciones1} iteraciones")
#else:
    #print(f"El método falló después de {iteraciones1} iteraciones con g1.")

# Ejecutamos el método del punto fijo para g2
raiz2, iteraciones2 = punto_fijo(g2, p0, TOL, N0)
if raiz2 is not None:
    print(f"Solución con g2: {raiz2} en {iteraciones2} iteraciones")
#else:
    #print(f"El método falló después de {iteraciones2} iteraciones con g2.")

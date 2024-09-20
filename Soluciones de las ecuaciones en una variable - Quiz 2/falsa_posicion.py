# Al igual que el método de la secante, este método no requiere el cálculo de la derivada, pero se basa en elegir dos puntos iniciales
# donde la función cambia de signo, similar al método de bisección.
# Sin embargo, el método de la posición falsaes generalmente más eficiente que la bisección, ya que no utiliza el punto medio,
# sino una línea secante entre los dos puntos.

def falsa_posicion(f, p0, p1, TOL, N0):
    q0 = f(p0)
    q1 = f(p1)
    i = 2  # Comenzamos en la segunda iteración

    while i <= N0:
        # Paso 3: Calcular la nueva aproximación
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        
        # Paso 4: Verificar si la aproximación es suficientemente precisa
        if abs(p - p1) < TOL:
            return p, i  # Retornamos la solución y el número de iteraciones
        
        # Paso 5: Incrementar el contador de iteraciones
        i += 1
        
        # Paso 6: Verificar en qué parte del intervalo está la raíz y actualizar
        q = f(p)
        if q * q1 < 0:  # La raíz está entre p y p1
            p0 = p1
            q0 = q1
        p1 = p
        q1 = q
    
    # Paso 8: Si no converge en N0 iteraciones
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

# Ejecutamos el método de Falsa Posición
raiz, iteraciones = falsa_posicion(f, p0, p1, TOL, N0)

if raiz is not None:
    print(f"La solución aproximada es: {raiz} en {iteraciones} iteraciones")
else:
    print(f"El método falló después de {iteraciones} iteraciones.")

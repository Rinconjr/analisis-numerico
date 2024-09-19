def steffensen(g, p0, TOL, N0):
    i = 1
    
    while i <= N0:
        # Paso 3: Calcular p1 = g(p0) y p2 = g(p1)
        p1 = g(p0)
        p2 = g(p1)
        
        # Aplicar la aceleración de Aitken
        p = p0 - ((p1 - p0) ** 2) / (p2 - 2 * p1 + p0)
        
        # Paso 4: Verificar si la aproximación es suficiente
        if abs(p - p0) < TOL:
            return p, i  # Procedimiento completado exitosamente
        
        # Paso 5: Incrementar el contador de iteraciones
        i += 1
        
        # Paso 6: Actualizar p0
        p0 = p
    
    # Paso 7: Si no converge dentro de N0 iteraciones
    print(f"El método falló después de {N0} iteraciones.")
    return None, i

# Ejemplo de uso:
# Definimos la función g(p)
def g(p):
    return (10 - 4*p**2)**(1/3)  # Ejemplo de g(p)

# Parámetros iniciales
p0 = 1.5  # Aproximación inicial
TOL = 1e-4  # Tolerancia
N0 = 100  # Número máximo de iteraciones

# Ejecutamos el método de Steffensen
raiz, iteraciones = steffensen(g, p0, TOL, N0)

if raiz is not None:
    print(f"La solución aproximada es: {raiz} en {iteraciones} iteraciones")
else:
    print(f"El método falló después de {iteraciones} iteraciones.")

def biseccion(f, a, b, TOL, N0):
    # Paso 1
    i = 1
    FA = f(a)
    
    # Paso 2
    while i <= N0:
        # Paso 3
        p = a + (b - a) / 2
        FP = f(p)
        
        # Paso 4
        if FP == 0 or (b - a) / 2 < TOL:
            return p  # Procedimiento completado exitosamente
        
        # Paso 5
        i += 1
        
        # Paso 6
        if FA * FP > 0:
            a = p
            FA = FP
        else:
            b = p
    
    # Paso 7
    print("El método fracasó después de N0 iteraciones.")
    return None


# Definimos la función f(x)
def funcion(x):
    return x**3 + 4*x**2 - 10 #Lo que se cambia es esto!!!!!

# Definimos el intervalo [a, b] #Lo que se cambia es esto!!!!!
a = 1
b = 2

# Definimos la tolerancia y el número máximo de iteraciones
TOL = 1e-5
N0 = 100

# Ejecutamos el método de bisección
raiz = biseccion(funcion, a, b, TOL, N0)

if raiz is not None:
    print(f"La raíz aproximada es: {raiz}")

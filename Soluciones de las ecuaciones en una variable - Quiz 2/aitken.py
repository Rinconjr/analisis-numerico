def aitken(p0, p1, p2):
    # Calcula la nueva aproximación acelerada usando la fórmula de Aitken
    numerator = (p1 - p0) ** 2
    denominator = p2 - 2 * p1 + p0
    
    if denominator == 0:
        print("División por cero en el método de Aitken")
        return None  # Si el denominador es cero, no se puede aplicar
    
    return p0 - numerator / denominator

# Ejemplo de uso
# Sucesión original (simulada)
p0 = 1.0
p1 = 1.5
p2 = 1.75

# Aplicar el método de Aitken
p_acelerado = aitken(p0, p1, p2)

if p_acelerado is not None:
    print(f"Valor acelerado: {p_acelerado}")
else:
    print("No se pudo aplicar el método de Aitken.")

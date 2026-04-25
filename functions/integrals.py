import numpy as np


def pesosNC(n) -> list[float]:
    # Calcula los pesos de la fórmula de Newton-Cotes de n puntos
    x = np.linspace(0, 1, n)
    A = np.ones((n, n))
    for i in range(1, n):
        A[i, :] = A[i-1, :] * x
    b = 1 / np.arange(1, n+1)
    w = np.linalg.solve(A, b)
    return w

def integralNC(f,a,b,n) -> float:
    w = pesosNC(n)
    x = np.linspace(a,b,n)
    y = f(x)
    Q = (b - a)*np.sum(y * w)
    return Q

def intNCcompuesta(f, a, b, L=50, n=3):
    z = np.linspace(a, b, L + 1)
    h = (b - a) / L
    w = pesosNC(n)
    Q = 0
    for i in range(L):
        x = np.linspace(z[i], z[i+1], n)
        y = f(x)  
        Q += h * np.sum(y * w)
    return Q

def intNCcompuestaAUTO(f, a, b, tol=1e-6, n=3) -> float:
    """
    Función que toma un valor de tolerancia y resuelve la integral con la precisión deseada.
    
    Parte de L=50 y, en cada iteración, duplica ese valor.
    """
    L = 50
    Qtemp = float("inf")
    while True:
        z = np.linspace(a, b, L + 1)
        h = (b - a) / L
        w = pesosNC(n)
        Q = 0
        for i in range(L):
            x = np.linspace(z[i], z[i+1], n)
            y = f(x)  
            Q += h * np.sum(y * w)
        if abs(Q-Qtemp)>tol:
            L*=2
            Qtemp = Q
        else:
            return Q

def ASRev(f,df,a,b,L,n):
    """
    A = ASRev(f,df,a,b,L,n)
    Calcula el Área de una superficie de revolución

    f: función
    df: derivada de f
    a,b: extremos de integración
    L: número de intervalos
    n: grado de integración
      n=2: Trapecio
      n=3: Simpson
    """
    
    g = lambda x: f[x]*np.sqrt(1 + df[x]**2)
    
    A = 2*np.pi()*intNCcompuesta(g,a,b,L,n)
    
    return A

def round_sig_figs(num, sig_figs) -> float:
    """
    Redondea un número a una cantidad específica de cifras significativas.
    """
    from math import floor, log10
    
    if num == 0:
        return 0
    
    # Calcular el orden de magnitud (potencia de 10)
    # Ej: 123.45 -> 10^2, 0.00123 -> 10^-3
    order_of_magnitude = floor(log10(abs(num)))
    
    # Calcular la posición del decimal para redondear
    # Ej: para 123.45 y 3 sig figs -> redondear a la unidad (0)
    # para 0.00123 y 2 sig figs -> redondear a la 5ta decimal (-5)
    decimal_places = sig_figs - 1 - order_of_magnitude
    
    return round(num, decimal_places)

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 21:13:24 2025

Chi cuadrado y otros
"""

def chi2(x:list, y:list, n:int, y_errors:list=None, weights:list=None, graf:int=0):
    # Paquetes
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.array(x)
    y = np.array(y)
    
    # Ajuste
    aux=0
    if (y_errors is None and weights is None):
        # ajuste sin pesos para estimar errores
        aux+=1
        pol = np.polyfit(x, y, n)
        y_pred = np.polyval(pol, x)
        residuals = y - y_pred
        y_errors = np.std(residuals, ddof=1) * np.ones_like(y)
    elif (y_errors is not None and weights is None):
        # ajuste con pesos estimados
        y_errors = np.array(y_errors)
        pol = np.polyfit(x, y, n, w=1/y_errors)
        y_pred = np.polyval(pol, x)
    elif (y_errors is None and weights is not None):
        # ajuste con pesos concretos
        weights = np.array(weights)
        pol = np.polyfit(x, y, n, w=weights)
        y_pred = np.polyval(pol, x)
        # Lo siguiente es dudoso, revisar
        residuals = y - y_pred
        y_errors = np.std(residuals, ddof=1) * np.ones_like(y)
    else:
        print("No puede ingresar tanto y_errors como weights. Ingrese solo uno de ellos, o ninguno.")
    
    chi = np.sum(((y-y_pred)/(y_errors))**2)
    nu = len(x) - (n+1)  # Grados de libertad (N - M)
    
    print(f"Chi-cuadrado: {chi:.4f}")
    print(f"Grados de libertad (nu): {nu}")
    if n==1:
        print(f"Parámetros del ajuste: pendiente = {pol[0]:.4f}, intersección = {pol[1]:.4f}")
    else:
        for i in pol:
            print(f"pol[{i}] = {pol[i]:.4f}")
    if aux == 1:
        print(f'Errores de y: {y_errors}')
    
    # Gráfico
    plt.errorbar(x, y, yerr=y_errors, fmt='o', label='Datos con errores', capsize=5)
    if n==1:
        plt.plot(x, y_pred, 'r-', label=f'Ajuste lineal: y = {pol[0]:.4f}x + {pol[1]:.4f}')
    elif n==2:
        plt.plot(x, y_pred, 'r-', label=f'Ajuste cuadrático: y = {pol[0]:.4f}x^2 + {pol[1]:.4f}x + {pol[2]:.4f}')
    else:
        plt.plot(x, y_pred, 'r-', label=f'Ajuste polinomial de grado {n}')
    plt.xlabel(input("Ingrese la etiqueta para el eje x: "))
    plt.ylabel(input("Ingrese la etiqueta para el eje y: "))
    plt.title(f'Ajuste lineal ($\chi^2$ = {chi:.2f}, $\\nu$ = {nu})')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    if aux==1:
        return chi, nu, pol, y_errors
    else:
        return chi, nu, pol



def chi2_lineal(data,grad=1,f=[],errors=0,weights=0):
    """
    Función que ajusta los datos mediante una regresión lineal y obtiene el valor de chi cuadrado para dicho modelo lineal.
    
    Argumentos
    ----------
        data: list of ndarrays
            Datos de X e Y y los errores de Y.
            Se asume que está ordenado de esta forma:
                data = [X, Y, sigY]
            El error de los datos Y es opcional. Si no son proporcionados, se debe indicar como parámetro adicional errors=1.
        grad: int, opcional
            Grado del polinomio de ajuste.
            Su valor por defecto es 1, lo que da un ajuste lineal.
        f: function, opcional
            Alternativa al parámetro grad.
            Se asume que es una función lambda de la forma:
                f = lambda x: [3**x, 2**x, np.ones_like(x)]
        errors: int, opcional
            1 si se incluyeron los errores de Y en el array "data".
            0 en el caso contrario.
            Su valor por defecto es 0.
        weights: int, opcional
            1 si se incluyeron pesos para cada valor de Y en el array "data".
            0 en el caso contrario.
            Su valor por defecto es 0.
    
    Retorna
    -------
        p: list
            Coeficientes del ajuste lineal realizado.
        covar: list
            Covarianza del ajuste.
            covar[0,0] es la covarianza de la pendiente
            covar[1,1] es la covarianza de la ordenada al origen
        sigY: int, solo si errors=0
            Errores estimados para que los datos de Y coincidan con el modelo lineal.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    
        # X, Y = np.array(X), np.array(Y)
        # # Paso 1: armado de la matriz M
        # M = np.array(f(X)).T
        # # Paso 2: armamos la matriz del sistema
        # A = M.T @ M
        # # Paso 3: armamos el lado derecho del sistema
        # b = M.T @ Y
        # # Paso 4: hallamos los coeficientes del polinomio que aproximan en el sentido de mínimos cuadrados
        # p = np.linalg.solve(A, b)
        # Yc = [p[n]*f.get_f(n)(X) for n in range(len(p))]
    # Ajuste
    if (errors == 0 and weights == 0):
        X, Y = data
        p, covar = np.polyfit(X,Y,1,cov=True)
        Yc = np.polyval(p, X)
        res = Y - Yc
        sigY = np.std(res, ddof=1) * np.ones_like(Y)
    elif (errors != 0 and weights == 0):
        X, Y, sigY = data
        w = np.abs(1/(sigY**2))
        p, covar = np.polyfit(X,Y,1,w=w,cov=True)
        Yc = np.polyval(p, X)
        res = Y - Yc
    elif (errors == 0 and weights != 0):
        return
    else:
        print("No puede ingresar tanto errores como pesos. Ingrese solo uno de ellos, o ninguno.")
    
    chi = np.sum((res/sigY)**2)
    nu = len(X) - 2
    
    # Informe
    print(f"Chi-cuadrado: {chi:.4f}")
    print(f"Grados de libertad (nu): {nu}")
    print(f"Parámetros del ajuste: pendiente = {p[0]:.4f} +/- {np.sqrt(covar[0,0]):.2f}, intersección = {p[1]:.4f} +/- {np.sqrt(covar[1,1]):.2f}")
    print(f'Errores de y: {sigY}')
    
    # Gráfico
    plt.errorbar(X, Y, yerr=sigY, fmt='o', label='Datos con errores', capsize=5)
    plt.plot(X, Yc, 'r-', label=f'Ajuste lineal: y = {p[0]:.4f}x + {p[1]:.4f}')
    plt.xlabel(input("Ingrese la etiqueta para el eje x: "))
    plt.ylabel(input("Ingrese la etiqueta para el eje y: "))
    plt.title(f'Ajuste lineal ($\chi^2$ = {chi:.2f}, $\\nu$ = {nu})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    if (errors == 0):
        return p, covar, sigY
    else:
        return p, covar


def chi2_no_lineal(f,data,LSR=0,errors=0,weights=0):
    """
    Función que calcula el valor de chi cuadrado para un modelo no lineal dado por el parámetro "f".
    
    Argumentos:
        f -- [function]
            Modelo no lineal.
            Se supone que la función es de la forma:
                def model(x):
                    return f(x)
        data -- [list of ndarrays]
            Datos de X e Y y los errores de Y.
            Se asume que está ordenado de esta forma: [X, Y, sigY].
            El error de los datos Y es opcional. Si no son proporcionados, se debe indicar como parámetro adicional errors=1.
        errors = 0 -- [int]
            1 si se incluyeron los errores de Y en el array "data".
            0 en el caso contrario.
            Su valor por defecto es 0.
    
    Retorna:
        p -- [list]
            Coeficientes del ajuste lineal realizado.
        covar -- [list]
            Covarianza del ajuste.
            covar[0,0] es la covarianza de la pendiente
            covar[1,1] es la covarianza de la ordenada al origen
        sigY -- [int] (solo cuando errors=0)
            Errores estimados para que los datos de Y coincidan con el modelo lineal.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Ajuste
    # if (errors == 0):
    #     X, Y = data
    #     X, Y = np.array(X), np.array(Y)
    #     Yc = f(X)
    #     res = Y - Yc
    #     sigY = np.std(res, ddof=1) * np.ones_like(Y)
    # elif (errors == 1):
    X, Y, sigY = data
    X, Y = np.array(X), np.array(Y)
    Yc = f(X)
    res = Y - Yc
    # else:
    #     print("No puede ingresar tanto errores como pesos. Ingrese solo uno de ellos, o ninguno.")
    
    chi = np.sum((res/sigY)**2)
    nu = len(X) - 2
    
    # Informe
    print(f"Chi-cuadrado: {chi:.4f}")
    print(f"Grados de libertad (nu): {nu}")
    # print(f"Parámetros del ajuste: pendiente = {p[0]:.4f} +/- {np.sqrt(covar[0,0]):.2f}, intersección = {p[1]:.4f} +/- {np.sqrt(covar[1,1]):.2f}")
    # print(f'Errores de y: {sigY}')
    
    # # Gráfico
    # plt.errorbar(X, Y, yerr=sigY, fmt='o', label='Datos con errores', capsize=5)
    # plt.plot(X, Yc, 'r-', label=f'Ajuste lineal: y = {p[0]:.4f}x + {p[1]:.4f}')
    # plt.xlabel(input("Ingrese la etiqueta para el eje x: "))
    # plt.ylabel(input("Ingrese la etiqueta para el eje y: "))
    # plt.title(f'Ajuste lineal ($\chi^2$ = {chi:.2f}, $\\nu$ = {nu})')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    if (errors == 0):
        return p, covar, sigY
    else:
        # return p, covar
        return chi, nu

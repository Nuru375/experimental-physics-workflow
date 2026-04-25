def puntofijo(f, p0, tol: float, max_iter: int = 100):
    # Convergencia del método:
    #   f(a)<=a & f(b)>=b
    #       o
    #   f(a)>=a & f(b)<=b

    iter: int = 1
    p: float = f(p0)
    while (abs(p-p0) > tol and iter < max_iter):
        p0 = p
        p = f(p)
        iter += 1
    
    if (iter == max_iter):
        print("Se superó el máximo de iteraciones.")
        return p, iter

    return p, iter


def bisec(f, inter: list[float], tol: float, max_iter: int = 100):
    
    a: float = inter[0]
    b: float = inter[1]
    m = (a+b)/2
    err_abs = abs(b-a)/2
    iter: int = 1

    while (err_abs/abs(m) > tol and iter < max_iter):
        if (f(a)*f(b) > 0):
            print("f(a) y f(b) tienen el mismo signo.")
            return [m, iter]
        
        if (f(a)*f(m)<0):
            b=m # a=a
        elif (f(m)*f(b)<0):
            a=m # b=b
        else: # f(m)=0
            break

        iter += 1
        err_abs /= 2
        m = a + err_abs

    if (iter == max_iter):
        print("Se superó el máximo de iteraciones.")
        return m, iter

    return m, iter


def newton(F, DF, p0, tol: float = 1e-8, max_iter: int = 100):
    from numpy.linalg import solve
    from numpy import add

    if (isinstance(p0, list)):
        a0, b0 = p0
        deltap = - solve(DF(a0, b0), F(a0, b0))
        a, b = add([a0, b0], deltap)

        iter: int = 1
        while (max(abs(deltap)) > tol and iter <= max_iter):
            deltap = - solve(DF(a, b), F(a, b))
            a, b = add([a, b], deltap)
            iter += 1

        if(iter > max_iter):
            print("se supero el maximo de iteraciones")

        residuo = F(a, b)

        return [a, b], iter, residuo
    else:
        p = p0 - F(p0)/DF(p0)

        iter: int = 1
        while (abs(p - p0) > tol and iter <= max_iter):
            p0 = p
            p = p0 - F(p0)/DF(p0)
            iter += 1

        if(iter > max_iter):
            print("se supero el maximo de iteraciones")

        residuo = F(p)

        return p, iter, residuo






# if __name__ == "__main__":
#     bisec()
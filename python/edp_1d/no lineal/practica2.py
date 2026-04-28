import time
from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
from matplotlib.pyplot import *

def punto_fijo(x0, xf, N, nu, ua, ub, fun, tol, uexacta=None):
    """Función que resuelve una EDO del tipo - u**2 - nu * u'' = f (t,u) usando un esquema de diferencias finitas de segundo
    orden y un método de punto fijo.
    ----------------
    x0:float --- Valor inciial del intervalo.
    xf:float --- Valor final del intervalo.
    N:int --- Número de particiones.
    nu:float --- Un número real positivo.
    ua:float --- Condición de contorno Dirichlet a la izquierda.
    ub:float --- Condición de contorno Dirichlet a la derecha.
    fun:callable --- Función que define la ecuación.
    tol:float --- Nivel de tolerancia para el test de parada del método iterativo.
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización.
    iter_max = 500  # Número máximo de iteraciones.
    t1 = time.time()
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    nu = float(nu)
    ua = float(ua)
    ub = float(ub)
    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    
    # Construimos la matriz D.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    D = nu/dx2 * D
    D[0, 0] = 1.0
    D[0, 1] = 0.0
    D[N, N] = 1.0
    D[N, N - 1] = 0.0
    D = D.tocsc()
    
    # Resolución.
    u_old = zeros(N+1)
    u_new = zeros(N+1)
    fun_x = fun(x)
    LU = splu(D)
    
    error = tol + 1    
    cont = 0  # Contador de iteraciones.
    while(error >= tol and cont <500):
        b = fun_x + u_old**2
        b[0] = ua
        b[-1] = ub
        u_new = LU.solve(b)
        error = max(abs(u_new - u_old))
        u_old = u_new
        
        cont += 1
        
    if(cont == 500):  # Avisamos si se alcanza el número máximo de iteraciones.
        print("Número máximo de iteraciones (", iter_max, ") alcanzado.")
        
    tf = time.time()  # Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    print("Número de iteraciones:", cont)
    plot(x, u_new, "b")
    if uexacta != None:  #Comprobamos si se ha pasado una solución exacta.
        plot(x, uexacta(x), "r")
        err = max(abs(u_new - uexacta(x)))
        print("Error cometido:", format(err))
        return err
    
def punto_fijo_bis(x0, xf, N, nu, ua, ub, fun, tol, uexacta=None):
    """Función que resuelve una EDO del tipo - u**2 - nu * u'' = f (t,u) usando un esquema de diferencias finitas de segundo
    orden y un método de punto fijo que añade el resultado de la iteración anterior a la diagonal.
    ----------------
    x0:float --- Valor inciial del intervalo.
    xf:float --- Valor final del intervalo.
    N:int --- Número de particiones.
    nu:float --- Un número real positivo.
    ua:float --- Condición de contorno Dirichlet a la izquierda.
    ub:float --- Condición de contorno Dirichlet a la derecha.
    fun:callable --- Función que define la ecuación.
    tol:float --- Nivel de tolerancia para el test de parada del método iterativo.
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización.
    iter_max = 500  # Número máximo de iteraciones.
    t1 = time.time()
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    nu = float(nu)
    ua = float(ua)
    ub = float(ub)
    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    # Construimos la matriz D.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    D = D.tocsc()
    R = lil_matrix((N + 1, N + 1), dtype="float64")
    R.setdiag(ones(N + 1), 0)
    R = R.tocsc()
    
    # Resolución.
    u_old = zeros(N+1)
    u_new = zeros(N+1)
    error = tol + 1
    fun_x = fun(x)
    fun_x[0] = ua
    fun_x[-1] = ub
    LU = splu(D)
    
    cont = 0  # Contador de iteraciones.
    while(error >= tol and cont <500):
        R.setdiag(u_old,0)
        A = nu/dx2 * D - R
        A[0, 0] = 1
        A[0, 1] = 0
        A[N, N] = 1
        A[N, N-1] = 0
        LU = splu(A)

        u_new = LU.solve(fun_x)
        error = max(abs(u_new - u_old))
        u_old = u_new
    
    cont += 1
        
    if(cont == 500):  # Avisamos si se alcanza el número máximo de iteraciones.
        print("Número máximo de iteraciones (", iter_max, ") alcanzado.")
        
        
    tf = time.time()  # Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    print("Número de iteraciones:", cont)
    plot(x, u_new, "b")
    if uexacta != None:  #Comprobamos si se ha pasado una solución exacta.
        plot(x, uexacta(x), "r")
        err = max(abs(u_new - uexacta(x)))
        print("Error cometido:", format(err))
        return err
    
def newton(x0, xf, N, nu, ua, ub, fun, tol, uexacta=None):
    """Función que resuelve una EDO del tipo - u**2 - nu * u'' = f (t,u) usando un esquema de diferencias finitas de segundo
    orden y el método de Newton.
    ----------------
    x0:float --- Valor inciial del intervalo.
    xf:float --- Valor final del intervalo.
    N:int --- Número de particiones.
    nu:float --- Un número real positivo.
    ua:float --- Condición de contorno Dirichlet a la izquierda.
    ub:float --- Condición de contorno Dirichlet a la derecha.
    fun:callable --- Función que define la ecuación.
    tol:float --- Nivel de tolerancia para el test de parada del método iterativo.
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización.
    iter_max = 500  # Número máximo de iteraciones.
    t1 = time.time()
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    nu = float(nu)
    ua = float(ua)
    ub = float(ub)
    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    # Construimos la matriz D.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    D = D.tocsc()
    U_diag = lil_matrix((N + 1, N + 1), dtype="float64")
    U_diag.setdiag(ones(N + 1), 0)
    U_diag = U_diag.tocsc()
    
    # Resolución.
    u_old = zeros(N+1)
    u_new = zeros(N+1)
    fun_x = fun(x)
    error = tol + 1
    
    
    cont = 0 # Contador de iteraciones.
    while(error >= tol and cont < iter_max):
        b = - nu/dx2 * D * u_old + u_old**2 + fun_x
        b[0] = - u_old[0] + ua
        b[-1] = - u_old[-1] + ub
        # Contruimos la matriz jacobiana.
        U_diag.setdiag(2 * u_old, 0)
        J = nu/dx2 * D - U_diag  # El diferencial de la función que describe el método iterativo.
        J[0, 0] = 1
        J[0, 1] = 0
        J[N, N] = 1
        J[N, N-1] = 0
        LU = splu(J)
        
        y_new = LU.solve(b)
        u_new = y_new + u_old
        error = max(abs(u_new - u_old))
        u_old = u_new
        
        cont += 1
        
    if(cont == 500):  # Avisamos si se alcanza el número máximo de iteraciones.
        print("Número máximo de iteraciones (", iter_max, ") alcanzado.")
        
        
    tf = time.time()  # Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    print("Número de iteraciones:", cont)
    plot(x, u_new, "b")
    if uexacta != None:  # Comprobamos si se ha pasado una solución exacta.
        plot(x, uexacta(x), "r")
        err = max(abs(u_new - uexacta(x)))
        print("Error cometido:", format(err))
        return err
    return tf - t1 
    
def calor_punto_fijo(x0, xf, t0, tf, N, M, nu, u0, ua, ub, fun, tol, uexacta=None):
    """Función que resuelve una EDO del tipo u_t - nu * u_xx = f (t,u) + u**2 usando un esquema de diferencias finitas de segundo
    orden en espacio, el método de Euler implícito en tiempo y un método de punto fijo.
    ----------------
    x0:float --- Valor inicial del intervalo.
    xf:float --- Valor final del intervalo.
    N:int --- Número de particiones en espacio.
    t0:float --- Valor inicial del intervalo.
    tf:float --- Valor final del intervalo.
    M:int --- Número de particiones en tiempo.
    nu:float --- Un número real positivo.
    ua:callable --- Condición de contorno Dirichlet a la izquierda.
    ub:callable --- Condición de contorno Dirichlet a la derecha.
    u0:callable --- Condición inicial del problema.
    fun:callable --- Función que define la ecuación.
    tol:float --- Nivel de tolerancia para el test de parada del método iterativo.
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización.
    iter_max = 500  # Número máximo de iteraciones.
    t1 = time.time()
    N = int(N)
    M = int(M)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    dt = (tf - t0) / float(M)
    nu = float(nu)
    x = linspace(x0, xf, N + 1)
    t = linspace(t0, tf, M + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")
    # Construimos la matriz A.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    D[0, 0] = 0.0
    D[0, 1] = 0.0
    D[N, N] = 0.0
    D[N, N - 1] = 0.0
    D = D.tocsc()
    A = Id + nu * dt/dx2 * D
    
    #Resolución.
    u0_x = u0(x)
    u_n = u0_x
    u_new = zeros(N+1)
    LU = splu(A)
    
    for n in range(M):
        cont = 0 #Contador de iteraciones.
        error = tol + 1
        u_old = u0_x
        
        while(error >= tol and cont <500):
            b = dt * fun(x,t[n+1]) + dt * u_old**2 + u_n 
            b[0] = ua(t[n+1])
            b[-1] = ub(t[n+1])
            
            u_new = LU.solve(b)
            error = max(abs(u_new - u_old))
            u_old = u_new
                
            cont += 1
        
        u_n = u_new
        
    if(cont == 500):  # Avisamos si se alcanza el número máximo de iteraciones.
        print("Número máximo de iteraciones (", iter_max, ") alcanzado.")
        
    tf = time.time()  # Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    print("Número de iteraciones:", cont)
    if uexacta != None:  #Comprobamos si se ha pasado una solución exacta.
        plot(x, u_new, "b", x, uexacta(x,t[-1]), "r")
        err = max(abs(u_new - uexacta(x,t[-1])))
        print("Error cometido:", format(err))
        return err    
    
################################## PRUEBAS ##################################

def f0(x):
    """Función que determina la ecuación diferencial."""
    y = cos(x) - cos(x)**2
    return y

def uexacta0(x):
    """Solución exacta del problema de contorno."""
    y = cos(x)
    return y

def f1(x, t):
    """Función en dos variables que determina la ecuación del calor."""
    y = -x * sin(x * t) + t**2 * cos(x * t) - cos(x * t)**2
    return y

def u0(x):
    """Solución exacta de la ecuación del calor."""
    return 1

def ua(t):
    """Condición de contorno izquierda de la ecuación del calor."""
    return 1

def ub(t):
    """Condición de contorno derecha de la ecuación del calor."""
    y = cos(pi * t)
    return y

def uexacta1(x, t):
    """Solución exacta de la ecuación del calor."""
    y = cos(x * t)
    return y
    
print("EJERCICIO 1")
mesh = [50,100,200,400]
error0 = 0
print("\n Versión uno Dirichlet ambos extremos------")

for N in mesh:
    error1 = punto_fijo(0, pi, N, 1, 1, -1, f0, 1e-8, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1   

print("\n EJERCICIO 2")
print("\n Versión dos Dirichlet ambos extremos------")

for N in mesh:
    error1 = punto_fijo_bis(0, pi, N, 1, 1, -1, f0, 1e-8, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1   

print("\n EJERCICIO 3")
print("\n Newton Dirichlet ambos extremos------")

for N in mesh:
    error1 = newton(0, pi, N, 1, 1, -1, f0, 1e-8, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1  

print("\n EJERCICIO 4")
print("\n Ecuación del calor, error espacial------")

mesh = [4,8,16,32,64]
for N in mesh:
    error1 = calor_punto_fijo(0, pi, 0 , 1, N, 1e6, 1, u0, ua, ub, f1, 1e-6, uexacta1)
    if N > 4:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 4, "el orden es = ---- \n")
    error0 = error1 
    
print("\n Ecuación del calor, error temporal------")

mesh = [4,8,16,32,64]
for M in mesh:
    error1 = calor_punto_fijo(0, pi, 0 , 1, 1e4, M, 1, u0, ua, ub, f1, 1e-6, uexacta1)
    if M > 4:
        print("Con M =", M, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con M =", 4, "el orden es = ---- \n")
    error0 = error1














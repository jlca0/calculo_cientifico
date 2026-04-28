import time
from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
from matplotlib.pyplot import *
    
def schwarz_2(x0, xf, N, l, nu, ua, ub, fun, izq, dch, tol, uexacta=None):
    """Función que resuelve una EDO del tipo u - alfa * u'' = f (t,u) usando un esquema de diferencias finitas de segundo
    orden y el método de descomposición de Schwarz. Se descompone el intervalo en dos subintervalos de longitud p1 y p2.
    ----------------
    x0:float --- Valor inciial del intervalo
    xf:float --- Valor final del intervalo
    N:int --- Número de particiones
    l:int --- Número de nodos en la intersección de ambos subintervalos.
    nu:float --- Un número real positivo
    ua:float --- Condición de contorno a la izquierda
    ub:float --- Condición de contorno a la derecha
    fun:callable --- Función que define la ecuación.
    izq:bool --- Si izq es TRUE la condición izquierda es tipo Neumann, si izq es False la condición izquierda es Dirichlet
    dch:bool --- Si dch es TRUE la condición derecha es tipo Neumann, si dch es False la condición derecha es Dirichlet
    tol: float --- Tolerancia al error del método.
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    iter_max = 500  # Número máximo de iteraciones.
    # Inicialización.
    t1 = time.time()
    N = int(N)
    l = int(l)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    nu = float(nu)
    ua = float(ua)
    ub = float(ub)
    izq = bool(izq)
    dch = bool(dch)
    x = linspace(x0, xf, N + 1)
    
    # Calculamos p1, p2 centrando el intervalo de l puntos en el de N+1 puntos.
    if (N%2 == 0):
        c = 0.5 * N
        if(l%2 == 0):
         p1 = c + 0.5 * l
         p2 = N - (p1 - l)
        else:
         p1 = c + 0.5 * (l-1) + 1
         p2 = N - (p1 - l)   
    else:
        c = 0.5 * (N+1)
        if(l%2 == 0):
         p1 = c + 0.5 * l
         p2 = N - (p1 - l)
        else:
         p1 = c + 0.5 * (l-1) + 1
         p2 = N - (p1 - l)   
    p1 = int(p1)
    p2 = int(p2)
    
    # Contruimos dos matrices A1, A2 para cada problema a resolver.
       
    D1 = lil_matrix((p1 + 1, p1 + 1), dtype="float64")
    Id1 = identity(p1 + 1, dtype="float64", format="csc")
    D1.setdiag(2.0 * ones(p1 + 1), 0)
    D1.setdiag(-1.0 * ones(p1), 1)
    D1.setdiag(-1.0 * ones(p1), -1)
    D1[p1, p1] = 0.0
    D1[p1, p1 - 1] = 0.0
    
    D2 = lil_matrix((p2 + 1, p2 + 1), dtype="float64")
    Id2 = identity(p2 + 1, dtype="float64", format="csc")
    D2.setdiag(2.0 * ones(p2 + 1), 0)
    D2.setdiag(-1.0 * ones(p2), 1)
    D2.setdiag(-1.0 * ones(p2), -1)
    D2[0, 0] = 0.0
    D2[0, 1] = 0.0

    # Clasificación
    if izq:  # Izquierda es Neumann
        # Modificamos D para imponer la primera ecuación.
        D1[0, 1] = 2 * D1[0, 1]
        
    else: #Izquierda no es Neumann.
        D1[0, 0] = 0.0
        D1[0, 1] = 0.0

    if dch:  # Derecha es Neumann.
        D2[p2, p2 - 1] = 2 * D2[p2, p2 - 1]

    else:  # Derecha no es Neumann.
        D2[p2, p2] = 0.0
        D2[p2, p2 - 1] = 0.0
        
    # Cambiamos del formato lil al formato csc por eficiencia.
    D1 = D1.tocsc()
    A1 = Id1 + nu / dx2 * D1  # La matriz A no es simetrica.
    D2 = D2.tocsc()
    A2 = Id2 + nu / dx2 * D2  # La matriz A no es simetrica.

    # Resolución.
    # Descomposición LU completa de una matriz Sparse.
    LU1 = splu(A1)
    LU2 = splu(A2)
    
    u_old_1 = ua + (ub - ua) * x[:p1+1] / (xf - x0)
    u_new_1 = zeros(p1 + 1)
    u_old_2 = ua + (ub - ua) * x[p1-l:] / (xf - x0)
    u_new_2 = zeros(p2 + 1)
    
    fun_x = fun(x)
    
    error = tol + 1
    cont = 0 # Contador de iteraciones.
    while(error >= tol and cont <iter_max):
        
        b1 = fun_x[:p1 + 1] # Bucle para (P1)
        b1[0] = int(not izq) * ua + int(izq) * (fun_x[0] - ua * 2 * nu / dx)
        b1[-1] = u_old_2[l]
        u_new_1 = LU1.solve(b1)
        
        b2 = fun_x[p1-l:] # Bucle para (P2)
        b2[0] = u_old_1[p1-l]
        b2[-1] = int(not dch) * ub + int(dch) * (fun_x[-1] - ub * 2 * nu / dx)
        u_new_2 = LU2.solve(b2)
        
        error = max(abs(u_new_1[(p1-l):] - u_new_2[:l+1]))
        u_old_1 = u_new_1
        u_old_2 = u_new_2
        
        cont += 1
        
    if(cont == iter_max):
        print("Número máximo de iteraciones (", iter_max, ") alcanzado.")
        
    tf = time.time()  # Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    print("Número de iteraciones:", cont)
    plot(x[:p1+1], u_new_1, "b")
    plot(x[p1-l:], u_new_2, "b")
    if uexacta != None:  #Comprobamos si se ha pasado una solución exacta.
        plot(x, uexacta(x), "r")
        err1 = max(abs(u_new_1 - uexacta(x[:p1+1])))
        err2 = max(abs(u_new_2 - uexacta(x[p1-l:])))
        err = max([err1, err2])
        print("Error cometido:", format(err))
        return err

def schwarz_3(x0, xf, N, l1, l2, nu, ua, ub, fun, izq, dch, tol, uexacta=None):
    """Función que resuelve una EDO del tipo u - alfa * u'' = f (t,u) usando un esquema de diferencias finitas de segundo
    orden y el método de descomposición de Schwarz. Se descompone el intervalo en tres subintervalos de longitud p1, p2 y p3.
    ----------------
    x0:float --- Valor inciial del intervalo
    xf:float --- Valor final del intervalo
    N:int --- Número de particiones
    l1:int --- Número de nodos en la intersección de los subintervalos I1 e I3.
    l2:int --- Número de nodos en la intersección de los subintervalos I3 e I2.
    nu:float --- Un número real positivo
    ua:float --- Condición de contorno a la izquierda
    ub:float --- Condición de contorno a la derecha
    fun:callable --- Función que define la ecuación.
    izq:bool --- Si izq es TRUE la condición izquierda es tipo Neumann, si izq es False la condición izquierda es Dirichlet
    dch:bool --- Si dch es TRUE la condición derecha es tipo Neumann, si dch es False la condición derecha es Dirichlet
    tol: float --- Tolerancia al error del método.
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    iter_max = 1000  # Número máximo de iteraciones.
    # Inicialización.
    t1 = time.time()
    N = int(N)
    l1 = int(l1)
    l2 = int(l2)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    nu = float(nu)
    ua = float(ua)
    ub = float(ub)
    izq = bool(izq)
    dch = bool(dch)
    x = linspace(x0, xf, N + 1)
    
    # Calculamos p1 centrando el intervalo de l1 puntos en 
    # el primer tercio del de N+1 puntos.
    c1 = (N + N%3)/3
    p1 = c1 + 0.5 * (l1 + l1%2) #  Extremo superior de I1.
    p31 = c1 - 0.5 * (l1 + l1%2) # Extremo inferior de I3.
        
    c2 = 2 * (N + N%3)/3
    p32 = c2 + 0.5 * (l2 + l2%2) #  Extremo superior de I3.
    p21 = c2 - 0.5 * (l2 + l2%2) #  Extremo inferior de I2.

    
    #La longitud de I1 coincide con p1.
    p1 = int(p1)
    #Calculamos p2 como la diferencia de p21 y N.
    p21 = int(p21)
    p2 = N - p21
    p31 = int(p31)
    p32 = int(p32)
    #Calculamos p3 como la diferencia de p31 y p32 que
    #da la longitud del intervalo interior.
    p3 = p32 - p31

    
    # Contruimos tres matrices A1, A2, A3 para cada problema a resolver.
       
    D1 = lil_matrix((p1 + 1, p1 + 1), dtype="float64")
    Id1 = identity(p1 + 1, dtype="float64", format="csc")
    D1.setdiag(2.0 * ones(p1 + 1), 0)
    D1.setdiag(-1.0 * ones(p1), 1)
    D1.setdiag(-1.0 * ones(p1), -1)
    D1[p1, p1] = 0.0
    D1[p1, p1 - 1] = 0.0
    
    D2 = lil_matrix((p2 + 1, p2 + 1), dtype="float64")
    Id2 = identity(p2 + 1, dtype="float64", format="csc")
    D2.setdiag(2.0 * ones(p2 + 1), 0)
    D2.setdiag(-1.0 * ones(p2), 1)
    D2.setdiag(-1.0 * ones(p2), -1)
    D2[0, 0] = 0.0
    D2[0, 1] = 0.0
    
    D3 = lil_matrix((p3 + 1, p3 + 1), dtype="float64")
    Id3 = identity(p3 + 1, dtype="float64", format="csc")
    D3.setdiag(2.0 * ones(p3 + 1), 0)
    D3.setdiag(-1.0 * ones(p3), 1)
    D3.setdiag(-1.0 * ones(p3), -1)
    D3[0, 0] = 0.0
    D3[0, 1] = 0.0
    D3[p3, p3] = 0.0
    D3[p3, p3 - 1] = 0.0


    # Clasificación
    if izq:  # Izquierda es Neumann
        # Modificamos D para imponer la primera ecuación.
        D1[0, 1] = 2 * D1[0, 1]
        
    else: #Izquierda no es Neumann.
        D1[0, 0] = 0.0
        D1[0, 1] = 0.0

    if dch:  # Derecha es Neumann.
        D2[p2, p2 - 1] = 2 * D2[p2, p2 - 1]

    else:  # Derecha no es Neumann.
        D2[p2, p2] = 0.0
        D2[p2, p2 - 1] = 0.0
        
    # Cambiamos del formato lil al formato csc por eficiencia.
    D1 = D1.tocsc()
    A1 = Id1 + nu / dx2 * D1  
    D2 = D2.tocsc()
    A2 = Id2 + nu / dx2 * D2  
    D3 = D3.tocsc()
    A3 = Id3 + nu / dx2 * D3  

    # Resolución.
    # Descomposición LU completa de una matriz Sparse.
    LU1 = splu(A1)
    LU2 = splu(A2)
    LU3 = splu(A3)
    
    u_old_1 = zeros(p1 + 1)
    u_new_1 = zeros(p1 + 1)
    u_old_2 = zeros(p2 + 1)
    u_new_2 = zeros(p2 + 1)
    u_old_3 = zeros(p3 + 1)
    u_new_3 = zeros(p3 + 1)
    
    fun_x = fun(x)
    
    error = tol + 1
    cont = 0 # Contador de iteraciones.
    while(error >= tol and cont <iter_max):
        
        b1 = fun_x[:p1 + 1] # Bucle para (P1)
        b1[0] = int(not izq) * ua + int(izq) * (fun_x[0] - ua * 2 * nu / dx)
        b1[-1] = u_old_3[l1]
        u_new_1 = LU1.solve(b1)
        
        b3 = fun_x[p31:p32+1]  # Bucle para (P1)
        b3[0] = u_old_1[p1 - l1]
        b3[-1] = u_old_2[l2]
        u_new_3 = LU3.solve(b3)
        
        b2 = fun_x[p21:]  # Bucle para (P2)
        b2[0] = u_old_3[p3-l2]
        b2[-1] = int(not dch) * ub + int(dch) * (fun_x[-1] - ub * 2 * nu / dx)
        u_new_2 = LU2.solve(b2)
        
        error1 = max(abs(u_new_1[(p1-l1):] - u_new_3[:l1+1]))
        error2 = max(abs(u_new_3[(p3-l2):] - u_new_2[:l2+1]))
        error = max([error1, error2])
        u_old_1 = u_new_1
        u_old_2 = u_new_2
        u_old_3 = u_new_3
        
        cont += 1
        
    if(cont == iter_max):
        print("Número máximo de iteraciones (", iter_max, ") alcanzado.")
        
    tf = time.time()  # Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    print("Número de iteraciones:", cont)
    plot(x[:p1+1], u_new_1, "b")
    plot(x[p31:p32+1], u_new_3, "b")
    plot(x[p21:], u_new_2, "b")
    if uexacta != None:  #Comprobamos si se ha pasado una solución exacta.
        plot(x, uexacta(x), "r")
        err1 = max(abs(u_new_1 - uexacta(x[:p1+1])))
        err3 = max(abs(u_new_3 - uexacta(x[p31:p32+1])))
        err2 = max(abs(u_new_2 - uexacta(x[p21:])))
        err = max([err1,err2,err3])
        print("Error cometido:", format(err))
        return err
    
def schwarz_k(x0, xf, N, l, nu, ua, ub, fun, izq, dch, tol, uexacta=None):
    """Función que resuelve una EDO del tipo u - alfa * u'' = f (t,u) usando un esquema de diferencias finitas de segundo
    orden y el método de descomposición de Schwarz. Se descompone el intervalo en k subintervalos de longitud p[i].
    ----------------
    x0:float --- Valor inciial del intervalo
    xf:float --- Valor final del intervalo
    N:int --- Número de particiones
    l:array([int]) --- Un vector con entrada l[i] correspondiente al número de nodos en la intersección de los subintervalos Ii e I(i+1).
    nu:float --- Un número real positivo
    ua:float --- Condición de contorno a la izquierda
    ub:float --- Condición de contorno a la derecha
    fun:callable --- Función que define la ecuación.
    izq:bool --- Si izq es TRUE la condición izquierda es tipo Neumann, si izq es False la condición izquierda es Dirichlet
    dch:bool --- Si dch es TRUE la condición derecha es tipo Neumann, si dch es False la condición derecha es Dirichlet
    tol: float --- Tolerancia al error del método.
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    iter_max = 1000  # Número máximo de iteraciones.
    # Inicialización.
    t1 = time.time()
    N = int(N)
    k = len(l)  # Número de solapamientos (en realidad coincide con k-1).
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    nu = float(nu)
    ua = float(ua)
    ub = float(ub)
    izq = bool(izq)
    dch = bool(dch)
    x = linspace(x0, xf, N + 1)  # Vector de nodos.
    q = []  # Lista de extremos de los subintervalos.
    p = []  # Lista de longitudes de los subintervalos.
    
    for i in range(1,k+1):
        q.append(i * (N + N%(k+1))/(k+1) - 0.5 * (l[i-1] + l[i-1]%2))
        q.append(i * (N + N%(k+1))/(k+1) + 0.5 * (l[i-1] + l[i-1]%2))
        
    q = [int(x) for x in q]  # Pasamos de float a int para usarlos como índices.

    p.append(q[1])  # La longitud de I1 coincide con q1.

    for i in range(1, k):
        p.append(q[2*i+1] - q[2*(i-1)])

    p.append(N - q[-2]) 
    
    p = [int(y) for y in p]
    
    # Contruimos una lista conteniendo las matrices para cada problema a resolver.
    A = []
    LU = []
    
    for i in range(k+1):
        D = lil_matrix((p[i] + 1, p[i] + 1), dtype="float64")
        A.append(D)
        # Construimos A[i] accediendo al último elemento de A.
        A[-1].setdiag(2.0 * ones(p[i] + 1), 0)
        A[-1].setdiag(-1.0 * ones(p[i]), 1)
        A[-1].setdiag(-1.0 * ones(p[i]), -1)
        A[-1][p[i], p[i]] = 0.0
        A[-1][p[i], p[i] - 1] = 0.0

        # Clasificación.
        if i == 0:  # Primer subdominio.
            if izq:  # Neumann a la izquierda.
                A[-1][0, 1] = 2 * A[-1][0, 1]
            else:  # Dirichlet a la izquierda.
                A[-1][0, 0] = 0.0
                A[-1][0, 1] = 0.0
            A[-1][p[i], p[i]] = 0.0
            A[-1][p[i], p[i] - 1] = 0.0
        
        elif i == k:  # Último subdominio.
            if dch:  # Neumann a la derecha.
                A[-1][p[i], p[i] - 1] = 2 * A[-1][p[i], p[i] - 1]
            else:  # Dirichlet a la derecha.
                A[-1][p[i], p[i]] = 0.0
                A[-1][p[i], p[i] - 1] = 0.0
            A[-1][0, 0] = 0.0
            A[-1][0, 1] = 0.0
            
        else:  # Subdominios intermedios.
            A[-1][0, 0] = 0.0
            A[-1][0, 1] = 0.0
            A[-1][p[i], p[i]] = 0.0
            A[-1][p[i], p[i] - 1] = 0.0
            
        Id = identity(p[i] + 1, dtype="float64", format="csc")
        # Pasamos a csc para eficiencia.
        A[-1].tocsc()
        A[-1] = Id + nu/dx2 * A[-1]
        
        # Hacemos la factorización LU.
        LU.append(splu(A[-1]))
 
    # Resolución.
    u_old = zeros(N+1) #Apilamos horizontalmente los vectores de cada problema.
    u_new = zeros(N+1)
  
    fun_x = fun(x)
    
    error = tol + 1
    cont = 0  # Contador de iteraciones.
    while(error >= tol and cont <iter_max):
        error_lista = []  # Vaciamos la lista de errores para esta iteración.
        b = fun_x.copy()  # Copiamos fun a b.
        
        # Resolvemos el primer intervalo.
        b[0] = int(not izq) * ua + int(izq) * (fun_x[0] - ua * 2 * nu / dx)
        b[q[1]] = u_old[q[1]]
        u_new[:q[1]+1] = LU[0].solve(b[:q[1]+1])
        error_lista.append(max(abs(u_new[:q[1]+1] - u_old[:q[1]+1])))
        u_old[:q[1]+1] = u_new[:q[1]+1]
        
        # Resolvemos los intervalos intermedios.
        for i in range(1, k):
            b[q[2*(i-1)]] = u_old[q[2*(i-1)]]
            b[q[2*i+1]] = u_old[q[2*i+1]]
            u_new[q[2*(i-1)]:q[2*i+1]+1] = LU[i].solve(b[q[2*(i-1)]:q[2*i+1]+1])
            u_old[q[2*(i-1)]:q[2*i+1]+1] = u_new[q[2*(i-1)]:q[2*i+1]+1]
        for i in range(1,k):
            error_lista.append(max(abs(u_new[q[2*(i-1)]:q[2*i+1]+1] - u_old[q[2*(i-1)]:q[2*i+1]+1])))
            
        # Resolvemos el último intervalo.
        b[q[-2]] = u_old[q[-2]]
        b[-2] = int(not dch) * ub + int(dch) * (fun_x[-1] - ub * 2 * nu / dx)
        u_new[q[-2]:] = LU[-1].solve(b[q[-2]:])
        error_lista.append(max(abs(u_new[q[-2]:] - u_old[q[-2]:])))
        u_old[q[-2]:] = u_new[q[-2]:]
            
        error = max(error_lista)
        
        cont += 1
        
    if(cont == iter_max):
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
    
###### PRUEBAS ######

def f0(x):
    """Función que define el problema de contorno."""
    y = 2 * sin(x)
    return y

def uexacta0(x):
    """Solución exacta del problema de contorno."""
    y = sin(x)
    return y

def f1(x):
    """Función que define el problema de contorno del ejercicio 2."""
    y = 5 * exp(-(x - 0.5)**2)
    return y

print("EJERCICIO 1.")
figure("Ejercicio 1.")
mesh = [50,100,200,400]
error0 = 0
for N in mesh:
    error1 = schwarz_2(0, pi, N, 10, 1, 0, 0, f0, False, False, 1e-8, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1
    
print("EJERCICIO 2.")
figure("Ejercicio 2.")
schwarz_2(0, 1, 500, 100, 2, 1, 0, f1, False, True, 1e-8)

print("\n")
print("EJERCICIO 3.")
mesh = [100,200,400]
error0 = 0
print("\n Versión 1 izq------")
figure("Ejercicio 3.")
for N in mesh:
    error1 = schwarz_3(0, pi, N, 10, 10, 1, 0, 0, f0, False, False, 1e-8, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1

print("EJERCICIO 4.")
mesh = [50,100,200,400]
error0 = 0
l = [5,5,5]
figure("Ejercicio 4.")
for N in mesh:
    error1 = schwarz_k(0, pi, N, l, 1, 0, 0, f0, False, False, 1e-8, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1




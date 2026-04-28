import time
from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
from matplotlib.pyplot import *

def contorno_directo_v1(x0, xf, N, alfa, ua, ub, fun, izq, dch, exacta=None):
    """Función que resuelve una EDO del tipo u - alfa * u'' = f (t,u) usando un esquema de diferencias finitas de segundo
    orden.
    ----------------
    x0:float --- Valor inciial del intervalo
    xf:float --- Valor final del intervalo
    N:int --- Número de particiones
    alfa:float --- Un número real positivo
    ua:float --- Condición de contorno a la izquierda
    ub:float --- Condición de contorno a la derecha
    fun:callable --- Función que define la ecuación.
    izq:bool --- Si izq es TRUE la condición izquierda es tipo Neumann, si izq es False la condición izquierda es Dirichlet
    dch:bool --- Si dch es TRUE la condición derecha es tipo Neumann, si dch es False la condición derecha es Dirichlet
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización.
    t1 = time.time()
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)
    izq = bool(izq)
    dch = bool(dch)
    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")
    # Construimos la matriz D, sin tener en cuenta la primera y la última ecuacion.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    # Construcción del vector b.
    b = fun(x)

    # Clasificación
    if izq:  # Izquierda es Neumann
        # Modificamos D para imponer la primera ecuación.
        D[0, 1] = 2 * D[0, 1]
        
        # Modificamos el primer elemento de b.
        b[0] -= ua * 2 * alfa / dx
    else: #Izquierda no es Neumann.
        D[0, 0] = 0.0
        D[0, 1] = 0.0
        
        b[0] = ua
    if dch:  # Derecha es Neumann.
        D[N, N - 1] = 2 * D[N, N - 1]

        b[N] += ub * 2 * alfa / dx
    else:  # Derecha no es Neumann.
        D[N, N] = 0.0
        D[N, N - 1] = 0.0
        
        b[N] = ub
        
    # Cambiamos del formato lil al formato csc por eficiencia.
    D = D.tocsc()
    A = Id + alfa / dx2 * D  # La matriz A no es simetrica.

    # Resolución.
    # Descomposición LU completa de una matriz Sparse.
    LU = splu(A)
    usol = LU.solve(b)

    tf = time.time()  # Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    plot(x, usol, "b")
    if exacta != None:  #Comprobamos si se ha pasado una solución exacta.
        plot(x, exacta(x), "r")
        err = max(abs(usol - exacta(x)))
        print("Error cometido:", format(err))
        return err

def contorno_directo_v2(x0, xf, N, alfa, ua, ub, fun, izq, dch, uexacta=None):
    """Función que resuelve una EDO del tipo u - alfa * u'' = f (t,u) usando un esquema de diferencias finitas de segundo
    orden, se aplica un método de penalización para operar con una matriz simétrica.
    ----------------
    x0:float --- Valor incial del intervalo
    xf:float --- Valor final del intervalo
    N:int --- Número de particiones
    alfa:float --- Un número real positivo
    ua:float --- Condición de contorno a la izquierda
    ub:float --- Condición de contorno a la derecha
    fun:callable --- Función que define la ecuación.
    izq:bool --- Si izq es TRUE la condición izquierda es tipo Neumann, si izq es False la condición izquierda es Dirichlet
    dch:bool --- Si dch es TRUE la condición derecha es tipo Neumann, si dch es False la condición derecha es Dirichlet
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización
    # Tomamos un número muy grande
    M = 1e30
    t1 = time.time()
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    alfa = float(alfa)
    ua = float(ua)
    ub = float(ub)
    izq = bool(izq)
    dch = bool(dch)
    x = linspace(x0, xf, N + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")
    # Construimos la matriz D, sin tener en cuenta la primera y la última ecuacion.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    D = D.tocsc()
    A = Id + alfa / dx2 * D
    # Construcción del vector b.
    b = fun(x)
        
    # Clasificación, no modificamos la estructura Sparse de A.
    if izq:  # Izquierda es Neumann.
        A[0, 0] = 0.5 * A[0, 0]
        # Modificamos el primer elemento de b.
        b[0] -= ua * alfa / dx
    else: #Izquierda no es Neumann.
        A[0, 0] = M
        
        b[0] = M * ua
        
    if dch:  # Derecha es Neumann.
        A[N, N] = 0.5 * A[N, N]

        b[N] += ub * alfa / dx
    else:  # Derecha no es Neumann.
        A[N, N] = M
        
        b[N] = M * ub

    # Resolución.
    # Descomposición LU completa de una matriz Sparse.
    LU = splu(A)
    usol = LU.solve(b)

    tf = time.time()  #Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    plot(x, usol, "b")
    if uexacta != None:  # Comprobamos si se ha pasado la solución exacta.
        plot(x, uexacta(x), "r")
        err = max(abs(usol - uexacta(x)))
        print("Error cometido:", format(err))
        return err

def lineas_implicito(x0, xf, t0, tf, u0, ua, ub, N, M, alfa,fuente, izq = False, dch = False, uexacta = None):
    """Función que resuelve la ecuación del calor u_t - alfa * u_xx = f (t,x) usando el método de líneas implícito.
    ----------------
    x0:float --- Valor inciial del intervalo
    xf:float --- Valor final del intervalo
    N:int --- Número de particiones en espacio.
    M:int --- Número de particiones en tiempo.
    alfa:float --- Un número real positivo
    ua:float --- Condición de contorno a la izquierda
    ub:float --- Condición de contorno a la derecha
    fun:callable --- Función que define la ecuación.
    izq:bool --- Si izq es TRUE la condición izquierda es tipo Neumann, si izq es False la condición izquierda es Dirichlet
    dch:bool --- Si dch es TRUE la condición derecha es tipo Neumann, si dch es False la condición derecha es Dirichlet
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización.
    t1 = time.time()
    N = int(N)
    M = int(M)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    dt = (tf - t0) / float(M)
    alfa = float(alfa)
    x = linspace(x0, xf, N + 1)
    t = linspace(t0, tf, M + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")
    # Construimos la matriz D.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    
    #Clasificación.
    if izq:  # Izquierda es Neumann.
        # Modificamos D para imponer la primera ecuación.
        D[0, 1] = 2 * D[0, 1]
        
        # Modificamos el primer elemento de b.
        ua_star = fuente(x[0],t) - 2 * alfa * dt/dx * ua(t)
    else: #Izquierda no es Neumann.
        D[0, 0] = 0.0
        D[0, 1] = 0.0
        
        ua_star = ua(t)
    if dch:  # Derecha es Neumann.
        D[N, N - 1] = 2 * D[N, N - 1]

        ub_star = fuente(x[0],t) + 2 * alfa * dt/dx * ub(t)
    else:  # Derecha no es Neumann.
        D[N, N] = 0.0
        D[N, N - 1] = 0.0
        
        ub_star = ub(t)
        
    # Pasamos de formato lil a csc, construimos A.
    D = D.tocsc()
    A = Id + alfa * dt/dx2 * D
    LU = splu(A)
    # Inicializamos el vector de nodos.
    usol = u0(x)
    cont = 0
    
    # Resolución.
    for n in range(M):
        b = dt * fuente(x,t[n+1]) + usol
        # Modificamos el primer y último elemento de b, si alguna es Neumann añadimos el elemento recursivo.
        b[0] = ua_star[n+1] + int(izq) * usol[0]
        b[N] = ub_star[n+1] + int(dch) * usol[-1]
        usol = LU.solve(b)
        # if(cont%10 == 0):
        #     plot(x, usol, "b", x, uexacta(x,t[n+1]), "r")
        #     pause(0.1)
        #     clf()
        # cont += 1

    tf = time.time()  #Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    plot(x, usol, "b")
    if uexacta != None:  #Comprobamos si se ha pasado la solución exacta.
        plot(x, uexacta(x,t[-1]), "r")
        err = max(abs(usol - uexacta(x,t[-1])))
        print("Error cometido:", format(err))
        return err   
    
def lineas_explicito(x0, xf, t0, tf, u0, ua, ub, N, alfa,fuente, izq = False, dch = False, uexacta = None):
    """Función que resuelve la ecuación del calor u_t - alfa * u_xx = f (t,x) usando el método de líneas explícito.
    ----------------
    x0:float --- Valor inciial del intervalo
    xf:float --- Valor final del intervalo
    N:int --- Número de particiones en espacio.
    alfa:float --- Un número real positivo
    ua:float --- Condición de contorno a la izquierda
    ub:float --- Condición de contorno a la derecha
    fun:callable --- Función que define la ecuación.
    izq:bool --- Si izq es TRUE la condición izquierda es tipo Neumann, si izq es False la condición izquierda es Dirichlet
    dch:bool --- Si dch es TRUE la condición derecha es tipo Neumann, si dch es False la condición derecha es Dirichlet
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización.
    t1 = time.time()
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    M = int(ceil(tf/dx2 * 2 * alfa))
    dt = (tf - t0) / float(M)
    alfa = float(alfa)
    x = linspace(x0, xf, N + 1)
    t = linspace(t0, tf, M + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")
    # Construimos la matriz D.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    
    #Clasificación.
    if izq:  # Izquierda es Neumann.
        # Modificamos D para imponer la primera ecuación.
        D[0, 1] = 2 * D[0, 1]
        
        # Modificamos el primer elemento de b.
        ua_star = - 2 * alfa * dt/dx * ua(t)
    else: # Izquierda no es Neumann.
        D[0, 0] = 0.0
        D[0, 1] = 0.0
        
        ua_star = ua(t)
    if dch:  # Derecha es Neumann.
        D[N, N - 1] = 2 * D[N, N - 1]

        ub_star = 2 * alfa * dt/dx * ub(t)
    else:  # Derecha no es Neumann.
        D[N, N] = 0.0
        D[N, N - 1] = 0.0
        
        ub_star = ub(t)
        
    # Pasamos de formato lil a csc, construimos A.
    D = D.tocsc()
    A = Id - alfa * dt/dx2 * D
    #Inicializamos el vector de nodos.
    usol = u0(x)
    cont = 0
    
    # Resolución.
    for n in range(M):
        usol = A * usol + dt * fuente(x,t[n]) 
        # Modificamos el primer y último elemento de b, si alguna es Neumann añadimos el elemento recursivo.
        usol[0] = int(izq) * usol[0] + ua_star[n+1]
        usol[-1] = int(dch) * usol[-1] + ub_star[n+1]
        # if(cont%50 == 0):
        #     plot(x, usol, "b", x, uexacta(x,t[n+1]), "r")
        #     pause(0.1)
        #     clf()
        # cont += 1

    tf = time.time()  #Comprobamos el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    plot(x, usol, "b")
    if uexacta != None:  #Comprobamos si se ha pasado la solución exacta.
        plot(x, uexacta(x,t[-1]), "r")
        err = max(abs(usol - uexacta(x,t[-1])))
        print("Error cometido:", format(err))
        return err
    
def theta_metodo(x0, xf, t0, tf, u0, ua, ub, N, M, alfa,fuente, theta, izq, dch, uexacta = None):
    """Función que resuelve la ecuación del calor u_t - alfa * u_xx = f (t,x) usando un theta método.
    ----------------
    x0:float --- Valor inciial del intervalo
    xf:float --- Valor final del intervalo
    N:int --- Número de particiones en espacio.
    alfa:float --- Un número real positivo
    ua:float --- Condición de contorno a la izquierda
    ub:float --- Condición de contorno a la derecha
    fun:callable --- Función que define la ecuación.
    theta:float --- El parámetro que determina el método. 
                    (Si theta es 0 es "explícito", si theta es 1 es implícito, si theta es 0.5 es Crank-Nicolson).
    izq:bool --- Si izq es TRUE la condición izquierda es tipo Neumann, si izq es False la condición izquierda es Dirichlet
    dch:bool --- Si dch es TRUE la condición derecha es tipo Neumann, si dch es False la condición derecha es Dirichlet
    uexacta:callable (opcional) --- La solución exacta del problema.
    ----------------
    """
    # Inicialización
    if (theta < 0 or theta > 1):  # Comprobamos que theta da una media ponderada.
        raise ValueError("Theta fuera del rango.")     
    t1 = time.time()
    N = int(N)
    x0 = float(x0)
    xf = float(xf)
    dx = (xf - x0) / float(N)
    dx2 = dx * dx
    if(theta < 0.5):  # Cuando el método es condicionalmente estable modificamos M.
        M = ceil(2 * (tf - t0) * (1 - 2 * theta) / dx2)
    M = int(M)
    dt = (tf - t0) / float(M)
    alfa = float(alfa)
    theta = float(theta)
    x = linspace(x0, xf, N + 1)
    t = linspace(t0, tf, M + 1)
    D = lil_matrix((N + 1, N + 1), dtype="float64")
    Id = identity(N + 1, dtype="float64", format="csc")
    # Construimos la matriz D.
    D.setdiag(2.0 * ones(N + 1), 0)
    D.setdiag(-1.0 * ones(N), 1)
    D.setdiag(-1.0 * ones(N), -1)
    
    #Clasificación.
    if izq:  # Izquierda es Neumann.
        # Modificamos D para imponer la primera ecuación.
        D[0, 1] = 2 * D[0, 1]

    else: #Izquierda no es Neumann.
        D[0, 0] = 0.0
        D[0, 1] = 0.0

    if dch:  # Derecha es Neumann.
        D[N, N - 1] = 2 * D[N, N - 1]

    else:  # Derecha no es Neumann.
        D[N, N] = 0.0
        D[N, N - 1] = 0.0

    D = D.tocsc()
    AE = Id - (1-theta) * alfa * dt/dx2 * D  # Matriz del método explícito.
    AI = Id + theta * alfa * dt/dx2 * D  # Matriz del método implícito.
    
    LU = splu(AI)
    
    #Resolución.
    #Inicializamos el vector de nodos.
    usol = u0(x)
    cont = 0  # Un contador por si queremos visualizar las soluciones.
    
    for n in range(M):
        b = dt * (theta * fuente(x,t[n+1]) + (1-theta) * fuente(x,t[n])) + AE * usol 
        # Modificamos el primer y último elemento de b según tengamos condiciones Neumann (True) o Dirchlet (False).
        b[0] = int(not izq) * ua(t[n+1]) + int(izq) * (b[0] - 2 * alfa * dt/dx * (theta * ua(t[n+1]) + (1 - theta) * ua(t[n]))) 
        b[N] = int(not dch) * ub(t[n+1]) + int(dch) * (b[N] + 2 * alfa * dt/dx * (theta * ub(t[n+1]) + (1 - theta) * ub(t[n])))  
        usol = LU.solve(b)
        plot(x, usol, "b")
        pause(0.1)
        clf()
        cont += 1
            
    tf = time.time()  # Comprobamos de nuevo el tiempo.
    print("Tiempo de ejecucion:", format(tf - t1))
    if (uexacta != None):  # Comprobamos si se ha pasado la solución exacta.
        plot(x,usol,'b',x,uexacta(x,t[-1]),'r')
        err = max(abs(usol - uexacta(x,t[-1])))  # Dibujamos la última iteración en tiempo.
        print("Error cometido:", format(err))
        return err
    
    
############ PRUEBAS ############


def f0(x):
    """Función que define el problema de contorno."""
    y = 2.0 * sin(x)
    return y

def uexacta0(x):
    """Solución exacta del problema de contorno."""
    y = sin(x)
    return y

def f1(x,t):
    """Función en dos variables que determina la ecuación del calor."""
    y = x * cos(x * t) + t**2 * sin(x * t)
    return y

def uexacta1(x,t):
    """Solución exacta de la ecuación del calor."""
    y = sin(x * t)
    return y

def ub(t):
    """Condición inicial Dirichlet de la ecuación del calor."""
    y = sin(t)
    return y

def ua(t):
    """Condición inicial Neumann de la ecuación del calor."""
    y = t
    return y

def u0(x):
    """Temperatura inicial en la ecuación del calor."""
    return 0

#Vectorizamos la función para poder hacer cálculo vectorizado con ella.
u0 = vectorize(u0)

    
print("EJERCICIO 1: CONTORNO DIRECTO.")
mesh = [50,100,200,400]
error0 = 0
print("\n Versión 1 izq------")

for N in mesh:
    error1 = contorno_directo_v1(0, pi, N, 1, 1, 0, f0, True, False, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", 0.5 * error0 / error1, "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1

print("\n Versión 2 izq------")

for N in mesh:
    error1 = contorno_directo_v2(0, pi, N, 1, 1, 0, f0, True, False, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", 0.5 * error0 / error1, "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1

print("\n Neumann en ambos extremos versión 2")

for N in mesh:
    error1 = contorno_directo_v2(0, pi, N, 1, 1, -1, f0, True, True, uexacta0)
    if N > 50:
        print("Con N =", N, "el orden es =", 0.5 * error0 / error1, "\n")
    else:
        print("Con N =", 50, "el orden es = ---- \n")
    error0 = error1

print("\n EJERCICIOs 3 Y 4: MÉTODO DE LÍNEAS.")

print("Neumann izquierda implícito. Orden en espacio.")
mesh = [5,10,20,40]
for N in mesh:
    error1 = lineas_implicito(0,1,0,1,u0,ua,ub,N, 1e4,1,f1,True, False, uexacta1)
    if N > 5:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 5, "el orden es = ---- \n")
    error0 = error1

print("Neumann izquierda explícito. Orden en espacio.")
for N in mesh:
    error1 = lineas_explicito(0,1,0,1,u0,ua,ub,N,1,f1,True, False, uexacta1)
    if N > 5:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 5, "el orden es = ---- \n")
    error0 = error1

print("Neumann izquierda Crank-Nicolson. Orden en tiempo.")
for M in mesh:
    error1 = theta_metodo(0,1,0,1,u0,ua,ub,1e4,M,1,f1,0.5,True, False, uexacta1)
    if M > 5:
        print("Con M =", M, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con M =", 5, "el orden es = ---- \n")
    error0 = error1

def c3(x):
    "Condición inicial en el ejercicio 3."
    y = 4 * x * (1-x)
    return y

def fun3(x, t):
    "Fuente en el ejercicio 3."
    y = 1 + np.cos(2 * np.pi * x)
    return y

a = 0
b = 1
c = 1
T = 0.5
N = 400
alpha = 0
gamma = 0
def contorno(t):
    return 0

figure(f'Ejercicio 3. Condición tipo Neumann: u(x,{T})')
grid(True)
xlabel('x')
ylabel(f'u(x,{T})')
x, u = theta_metodo(a,b,0,T,c3,contorno,contorno,N,N,c,fun3,0.5,False,True)
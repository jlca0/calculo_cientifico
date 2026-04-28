import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity

def exacta_a(x):
    """Solucion exacta del partado A"""
    return np.sin(2.*np.pi*x)/(1+4*np.pi**2)

def funs_a(x):
    """Funciones p(x), q(x) y f(x)"""
    p = np.zeros(len(x))
    q = np.ones(len(x))
    r = -np.sin(2.*np.pi*x)
    return (p, q, r)

def diffin(a,b,alpha,beta,funs,N):
    """Resuelve numéricamente, aplicando un esquema d ediferencias ifnitas de segundo orden, la ecuación diferencial lineal 
    y'' = p*y' + q*y + r. A izquierda y derecha se imponen condiciones de contorno tipo Dirichlet.
    
    Entrada:
        a: (float) Extremo inferior del intervalo.
        b: (float) Extremo superior del intervalo.
        alpha: (float) Condición Dirichlet a izquierda.
        beta: (float) Condición Dirichlet a derecha.
        funs: (callable) Array que contiene las funciones en la forma array([p, q, r]).
        N: (int) Número de particiones.
    
    Salida:
        x: (np.ndarray) Discretización del intervalo.
        U: (np.ndarray) Array de soluciones aproximadas.
    """
    h = (b-a)/(N +1)# paso de malla
    x = np.linspace(a, b, N+2) # discretizacion del intervalo [a, b]
    (p, q, r) = funs(x) # coeficientes de la ecuacion
    A = lil_matrix((N,N)) # matriz del sistema en la clase lil (Linked List Matrix) de Scipy
    for i in range(N-1):
        A[i, i] = 1.+0.5*q[i+1]*h**2 # diagonal
        A[i, i+1] = -0.5*(1.-0.5*p[i+1]*h) # superdiagonal
        A[i+1, i] = -0.5*(1.+0.5*p[i+2]*h) # subdiagonal
    A[N-1, N-1] = 1.+0.5*q[N]*h**2 # ultimo elemento de la diagonal
    F = -h**2/2*r[1:N+1]

# modificacion del termino independiente
    F[0] += alpha*0.5*(1.+0.5*p[1]*h)
    F[N-1] += beta*0.5*(1.-0.5*p[N]*h)
    
# resolucion del sistema
    
    A = A.tocsc() #pone la matriz en formato csc (Compressed Sparse Column)
    
    LU = splu(A)  #descomposicion LU

    U = LU.solve(F) # solucion del sistema lineal

# condiciones de contorno
    U = np.append(alpha, U)
    U = np.append(U, beta)
    return x,U


# Datos del problema
a = 0. # extremo inferior del intervalo
b = 1. # extremo superior del intervalo
alpha = 0. # condicion de contorno en a
beta = 0. # condicion de contorno en b
N = 10 # numero de particiones (m=N-1)

tini = perf_counter()

x,U = diffin(a,b,alpha, beta, funs_a, N)

tfin = perf_counter()

Ue = exacta_a(x) # solucion exacta

error = max(abs(Ue-U)) # error cometido

# Resultados
print('-----')
print('Tiempo CPU: ' + str(tfin-tini))
print('Error: ' + str(error))
print('Paso de malla: ' + str((b-a)/(N+1)))
print('----- \n')

# solucion exacta con mayor resolucion
xx = np.linspace(a, b, 500)
Ue = exacta_a(xx)

plt.figure('Ejercicio A')
plt.plot(x, U, 'b-o') # dibuja la solucion aproximada
plt.plot(xx, Ue,'r') # dibuja la solucion exacta
plt.legend(['aproximada', 'exacta'], loc='lower left')
plt.grid(True)
plt.show()

######### EJERCICIO A #########
print('EJERCICIO A \n')
malla = [20,40,80,160,320,640]
error0 = 1
for K in malla:
    x,U = diffin(a,b,alpha, beta, funs_a, K)
    error1 = max(abs(U-exacta_a(x)))
    if K == 20:
        print(f'El orden del método con N = {K} es: ---- \n')
    else:
        print(f'El orden del método con N = {K} es: {(np.log(error0)-np.log(error1))/np.log(2)} \n')
    error0 = error1

######### EJERCICIO B #########
print('EJERCICIO B \n')

def funs_b(x):
    """Funciones p(x), q(x) y f(x)"""
    p = -np.ones(len(x))
    q = np.ones(len(x))
    r = np.cos(np.pi*x)
    return (p, q, r)

# Datos del problema
a = 0. # extremo inferior del intervalo
b = 2. # extremo superior del intervalo
alpha = 1. # condicion de contorno en a
beta = 1. # condicion de contorno en b
N = 100 # numero de particiones (m=N-1)

tini = perf_counter()

x,U = diffin(a,b,alpha, beta, funs_b, N)

tfin = perf_counter()

# Resultados
print('-----')
print('Tiempo CPU: ' + str(tfin-tini))
print('Paso de malla: ' + str((b-a)/(N+1)))
print('----- \n')

plt.figure('Ejercicio B')
plt.plot(x, U, 'b') # dibuja la solucion aproximada
plt.grid(True)
plt.show()

######### EJERCICIO C #########
print('EJERCICIO C')

def diffin_neumann(a,b,gamma,beta,funs,N):
    """Resuelve numéricamente, aplicando un esquema de diferencias finitas de segundo orden, la ecuación diferencial lineal 
    y'' = p*y' + q*y + r. A izquierda se impone una condición Neumann y a derecha una condición Neumann.
    
    Entrada:
        a: (float) Extremo inferior del intervalo.
        b: (float) Extremo superior del intervalo.
        gamma: (float) Condición Neumann a izquierda.
        beta: (float) Condición Dirichlet a derecha.
        funs: (callable) Array que contiene las funciones en la forma array([p, q, r]).
        N: (int) Número de particiones.
    
    Salida:
        x: (np.ndarray) Discretización del intervalo.
        U: (np.ndarray) Array de soluciones aproximadas.
    """
    h = (b-a)/(N + 1)# paso de malla
    x = np.linspace(a, b, N+2) # discretizacion del intervalo [a, b]
    (p, q, r) = funs(x) # coeficientes de la ecuacion
    A = lil_matrix((N+1,N+1)) # matriz del sistema en la clase lil (Linked List Matrix) de Scipy
    A[0, 0] = 1.+0.5*q[0]*h**2
    A[0, 1] = -1.
    A[1, 0] = -0.5*(1+0.5*p[1]*h)
    for i in range(1,N):
        A[i, i] = 1.+0.5*q[i]*h**2 # diagonal
        A[i, i+1] = -0.5*(1.-0.5*p[i]*h) # superdiagonal
        A[i+1, i] = -0.5*(1.+0.5*p[i+1]*h) # subdiagonal
    A[N, N] = 1.+0.5*q[N]*h**2 # ultimo elemento de la diagonal
    F = -h**2/2*r[:N+1]

# modificacion del termino independiente
    F[0] -= 2*(0.5*(1.+0.5*p[0]*h))*h*gamma
    F[N] += beta*0.5*(1.-0.5*p[N]*h)
    
# resolucion del sistema
    
    A = A.tocsc() #pone la matriz en formato csc (Compressed Sparse Column)
    
    LU = splu(A)  #descomposicion LU

    U = LU.solve(F) # solucion del sistema lineal

# condiciones de contorno
    U = np.append(U, beta)
    return x,U

def funs_c(x):
    """Funciones p(x), q(x) y f(x)"""
    p = np.zeros(len(x))
    q = np.ones(len(x))
    r = -np.sin(2*np.pi*x)
    return (p, q, r)

def exacta_c(x):
    """Solución exacta para el apartado C"""
    y = np.sin(2.*np.pi*x)/(1+4*np.pi**2) + (1-2*np.pi+4*np.pi**2)/((1+4*np.pi**2)*(1+np.e**2)) * (np.exp(x) - np.exp(2-x))
    return y

# Datos del problema
a = 0. # extremo inferior del intervalo
b = 1. # extremo superior del intervalo
alpha = 1. # condicion de contorno en a
beta = 0. # condicion de contorno en b
N = 10 # numero de particiones (m=N-1)

tini = perf_counter()

x,U = diffin_neumann(a,b,alpha, beta, funs_c, N)

tfin = perf_counter()

Ue = exacta_c(x) # solucion exacta

error = max(abs(Ue-U)) # error cometido

# Resultados
print('-----')
print('Tiempo CPU: ' + str(tfin-tini))
print('Error: ' + str(error))
print('Paso de malla: ' + str((b-a)/(N+1)))
print('----- \n')

# solucion exacta con mayor resolucion
xx = np.linspace(a, b, 500)
Ue = exacta_c(xx)

plt.figure('Ejercicio C')
plt.plot(x, U, 'b-o') # dibuja la solucion aproximada
plt.plot(xx, Ue,'r') # dibuja la solucion exacta
plt.legend(['aproximada', 'exacta'], loc='lower left')
plt.grid(True)
plt.show()

malla = [20,40,80,160,320,640]
error0 = 1
for K in malla:
    x,U = diffin_neumann(a,b,alpha, beta, funs_c, K)
    error1 = max(abs(U-exacta_c(x)))
    if K == 20:
        print(f'El orden del método con N = {K} es: ---- \n')
    else:
        print(f'El orden del método con N = {K} es: {(np.log(error0)-np.log(error1))/np.log(2)} \n')
    error0 = error1

######### EJERCICIO D #########
print('EJERCICIO D')

def funs_d(x):
    """Funciones p(x), q(x) y f(x)"""
    p = - 2 * x / (1 + x**2)
    q = 1 / (1 + x**2)
    r = x**2 / (1 + x**2)
    return (p, q, r)

# Datos del problema
a = 0. # extremo inferior del intervalo
b = 1. # extremo superior del intervalo
alpha = 1. # condicion de contorno en a
beta = 0. # condicion de contorno en b
N = 100 # numero de particiones (m=N-1)

tini = perf_counter()

x,U = diffin_neumann(a,b,alpha, beta, funs_d, N)

tfin = perf_counter()

# Resultados
print('-----')
print('Tiempo CPU: ' + str(tfin-tini))
print('Paso de malla: ' + str((b-a)/(N+1)))
print('----- \n')

plt.figure('Ejercicio D')
plt.plot(x, U, 'b') # dibuja la solucion aproximada
plt.legend(['aproximada'])
plt.grid(True)
plt.show()
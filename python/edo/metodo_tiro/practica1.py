import numpy as np
import matplotlib.pyplot as plt 
import time

######### EJERCICIO 1 #########
print('EJERCICIO 1 \n')

def _set_mesh(a,b,N,y0):
    h = (b-a)/N # mesh size
    t = np.zeros(N+1) # node vector
    y = np.zeros([len(y0),N+1])
    t[0] = a # initial time
    y[:,0] = y0

    return h, t, y


def rk4(a, b, fun, N, y0):
    """Applies the RK4 method to the equation y' = f(t,y) in the interval [a, b]
    with N partitions and initial condition y0. 
    
    Arguments:
        a:  (float)   Lower bound of the interval.
        b:  (float)   Upper bound of the interval.
        fun: (Callable) Function which defines the equation.
        N: (int) Number of partitions.
        y0: (np.ndarray) Initial condition.
    
    Salida:
        y: (np.ndarray) Solution matrix.
    """
    
    h, t, y = _set_mesh(a, b, N, y0)
    
    # RK4 method
    for k in range(N):
        k1 = fun(t[k], y[:,k]) 
        k2 = fun(t[k] + h/2, y[:,k] + (h/2)*k1)
        k3 = fun(t[k] + h/2, y[:,k] + (h/2)*k2)
        k4 = fun(t[k] + h, y[:,k] + h*k3)
        y[:,k+1] = y[:,k]+(h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t[k+1] = t[k]+h
    
    return y

def _not_same_sign(a: float, b: float):
    return a*b < 0

def bipartition(fun,a,b,tol):
    """Assuming a sign change, it solves fun = 0 in the
    interval [a,b] making use of the bipartition method for a certain
    tolerance to error tol.
    
    Arguments:
        a:  (float)   Lower bound of the interval.
        b:  (float)   Upper bound of the interval.
        fun: (Callable) Function which defines the equation.
        tol: (float) Tolerance to error.

    Salida:
        cn: (float) Approximated solution.
    """
    an = a
    bn = b
    fan = fun(a)
    fbn = fun(b)

    error = tol + 1

    while error > tol:
        cn = 0.5*(an + bn)
        fcn = fun(cn)
        if _not_same_sign(fan,fcn):
            bn = cn
            fbn = fcn
        else:
            an = cn
            fan = fcn
        error = abs(0.5*(bn - an))
    return cn

def shooting_function(v, a, b, alfa, beta, N, fun):
    """It evaluates in b, using the RK4 method, the function defined by the
    associated IVP for the BVP, then substracts beta.
    
    Arguments:
        v: (float) Initial condition (speed).
        a:  (float)   Lower bound of the interval.
        b:  (float)   Upper bound of the interval.
        alfa: (float) Left initial condition for the BVP.
        beta: (float) Right initial condition for the BVP.
        N: (int) Number of partitions for the RK4 method.
        fun: (Callable) Function which defines the problem.
    
    Salida:
        (float) Value of the function given by the PVI in b minus beta.
    """
    y0 = np.zeros(2)
    y0[0] = alfa
    y0[1] = v
    y = rk4(a,b,fun,N,y0)
    return y[0,-1] - beta 
    
def shooting_method_bi(a, b, y0, N, fun, tol, v0, v1):
    """Solves the PVI applying the shooting method together with the
    bipartition method.
    
    Argumentos:
        a:  (float)   Lower bound of the interval.
        b:  (float)   Upper bound of the interval.
        y0: (np.ndarray) Initial condition.
        N: (int) Number of partitions.
        fun: (callable) Function which defines the problem.
        tol: (float) Tolerance to error for the bipartition method.
        v0: (float) First initial guess for the bipartition method.
        v1: (float) Second initial guess for the bipartition method.

    Salida:
        t: (np.ndarray) Node vector.
        y: (np.ndarray) Solution vector.
    """
    # Añadimos los parámetros a F
    def F(v):
        return shooting_function(v, a, b, y0[0], y0[-1], N, fun)
    # Usamos dicotomía para calcular el ángulo de tiro v
    v_sol = bipartition(F,v0, v1, tol)
    # Resolvemos el PVI "equivalente" al PC
    y0 = np.array([a, v_sol])
    y = rk4(0, np.pi, f, 200, y0)
    t = np.linspace(0, np.pi, 201)
    return t, y

######### EJERCICIO 2 #########
print('EJERCICIO 2 \n')

# Tanteamos f para buscar un intervalo con un cero

v_test = np.linspace(0,25,10)

def f(t,x):
    """Función que define el problema."""
    f1 = x[1] 
    f2 = 2*np.cos(t)*x[1]+0.01*x[0]
    return np.array([f1,f2])

lista = []
for k in range(len(v_test)):
    lista += [funcion_tiro(v_test[k],0,np.pi,0,25, 200,f)]
print(f'Valores de f en [0,25]: {lista} \n')
print('Vemos que parece haber un cambio de signo entre 0 y  2.5 (= 0 + 25/100) \n')

# Una vez identificado el cambio de signo, llamamos al método
t0 = time.time()
t_dic, y_dic = metodo_tiro_dic(0, np.pi, [0, 25], 200, f, 1e-12, 0, 2.5)
t1 = time.time()
print(f'Tiempo de ejecución dicotomía: {t1-t0} \n')
plt.figure('Método del tiro - RK4 + Dicotomía')
plt.plot(t_dic, y_dic[0,:])
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Método del tiro - RK4 + Dicotomía')

######### EJERCICIO 3 #########
print('EJERCICIO 3 \n')

def secant(fun,a,b,tol,iter_max):
    """It solves fun = 0 in the interval [a,b] making use of the secant 
    method for a certain tolerance to error tol.
    
    Arguments:
        a:  (float)   Lower bound of the interval.
        b:  (float)   Upper bound of the interval.
        fun: (Callable) Function which defines the equation.
        tol: (float) Tolerance to error.
        iter_max: (int) Maximum number of iterations.

    Salida:
        cn: (float) Approximated solution.
    """
    an = a
    bn = b
    fan=fun(a)
    fbn=fun(b)
    
    # Comprobamos si ya tenemos un cero.
    if fan==0:
        return a
    elif fbn==0:
        return b

    error = tol + 1 
    cont = 0   

    while error>tol and cont<iter_max:
        if fan==fbn:
            print('Secant method: Cannot be applied.')
            return None
    
        cn = bn - ((bn-an)/(fbn-fan))*fbn
        fcn = fun(cn)
        error = abs(bn-an)  # actualizamos el error
        cont += 1  # actualizamos el contador

        if fcn==0:
            return cn
        
        an = bn
        bn = cn
        fan = fbn
        fbn = fcn

    if cont == iter_max:
        print('Metodo secante: Se ha alcanzado iter_max') 

    return cn

def shooting_method_sec(a, b, y0, N, fun, tol, v0, v1):
    """Solves the PVI applying the shooting method together with the
    secant method.
    
    Argumentos:
        a:  (float)   Lower bound of the interval.
        b:  (float)   Upper bound of the interval.
        y0: (np.ndarray) Initial condition.
        N: (int) Number of partitions.
        fun: (callable) Function which defines the problem.
        tol: (float) Tolerance to error for the secant method.
        v0: (float) First initial guess for the secant method.
        v1: (float) Second initial guess for the secant method.

    Salida:
        t: (np.ndarray) Node vector.
        y: (np.ndarray) Solution vector.
    """
    # Añadimos los parámetros a F
    def F(v):
        return shooting_function(v, a, b, y0[0], y0[-1], N, fun)
    # Usamos secante
    v_sol = secant(F,v0, v1, tol, 100)
    # Resolvemos el PVI "equivalente" al PC
    y0 = np.array([a, v_sol])
    y = rk4(0, np.pi, f, 200, y0)
    t = np.linspace(0, np.pi, 201)
    return t, y

t0 = time.time()
t_sec, y_sec = metodo_tiro_sec(0, np.pi, [0, 25], 200, f, 1e-12, 1, 2.5)
t1 = time.time()
print(f'Tiempo de ejecución secante: {t1-t0} \n')

# Hacemos la grafica
plt.figure('Método del tiro - RK4 + Secante')
plt.plot(t_sec, y_sec[0,:])
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Método del tiro - RK4 + Secante')
plt.show()

# Comparamos los tiempos y resultados obtenidos.
print(f'Error total entre dicotomía y secante: {abs(np.max(y_sec-y_dic))} \n')
print('Observación: Mientras que obtenemos resultados prácticamente iguales el tiempo de ejecución cuando usamos el método de la secante es muy inferior. Eso se debe a que estamos buscando el cero de una función afín y, en ese caso, el método es exacto.')

def afin(): 
    """Pinta la gráfica de la función tiro a la que buscamos un cero. Sirve también para tantear."""
    def F(v):
        return funcion_tiro(v, 0, np.pi, 0, 25, 200, f)
    F = np.vectorize(F)
    plt.figure('Función tiro')
    plt.plot(v_test,F(v_test))
    plt.xlabel('v')
    plt.ylabel('F(v) = y_v - beta')
    plt.title('Función tiro a la que buscamos un cero')
    plt.show()
afin()
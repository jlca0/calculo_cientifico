import time
from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
from matplotlib.pyplot import *

def punto_fijo_1(xi,xf,Nx,yi,yf,Ny,ti,tf,M,nu,u0,u1,u2,u3,fun,g,tol):  
    """Resuelve una ecuación del tipo u - nu * \Delta u = f, donde u es una función de
    dos variables, aplicando un esquema de diferencias finitas de segundo orden y construye
    una matriz no simétrica.
    ----------------
    xi:float --- Valor inciial del intervalo en el eje X.
    xf:float --- Valor final del intervalo en el eje X.
    Nx:int --- Número de particiones en el eje X.
    yi:float --- Valor inciial del intervalo en el eje Y.
    yf:float --- Valor final del intervalo en el eje Y.
    Ny:int --- Número de particiones en el eje Y.
    nu:float --- Un número real positivo
    u0:float --- Condición de contorno abajo.
    u1:float --- Condición de contorno a la izquierda.
    u2:float --- Condición de contorno arriba.
    u3:float --- Condición de contorno a la derecha.
    fuente:callable --- Función que define la ecuación.
    ----------------
    """
    iter_max = 500 # Número máximo de iteraciones.
    Nx=int(Nx)
    Ny=int(Ny)
    M = int(M)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)   
    dt = (tf - ti)/int(M)
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    t = linspace(ti,tf,M+1)
    X,Y=meshgrid(x,y) 
    Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx),1)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx),-1)
    My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)
       
    Mx[0,0]=0.0
    Mx[0,1]=0.0
    Mx[Nx,Nx]=0.0
    Mx[Nx,Nx-1]=0.0
        
    My[0,0]=0.0
    My[Nx,Nx]=0.0
    
    for i in range(1,Ny): 
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx   
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My 
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    
    
    A=Id+nu*dt*A
    A=A.tocsc()
    LU = splu(A)
    
    g0 = g(X,Y)
    u_k = g0
    u_new = zeros(N)
    
    for k in range(M):
        cont = 0 #Contador de iteraciones.
        error = tol + 1
        u_old = g0
        
        while(error >= tol and cont < iter_max):
            b = dt * fun(X,Y,t[k+1]) - dt * u_old**2 + u_k 
            b[0,:]=u0(x,t[k+1])
            b[Ny,:]=u2(x,t[k+1])            
            b[:,0]=u3(y,t[k+1])
            b[:,Nx]=u1(y,t[k+1])
            
            u_new = LU.solve(b.reshape(N))
            error = max(abs(u_new - u_old.reshape(N)))
            u_old = u_new.reshape((Nx+1,Ny+1))
                
            cont += 1
        
        u_k = u_new.reshape((Nx+1,Ny+1))
            
            
    if(cont == iter_max):
        print("Se ha alcanzado el número máximo de iteraciones:",iter_max)
    else:
        print("Número de iteraciones:",cont)
        
    cu=contourf(X,Y,u_k,20)
    colorbar(cu)
    cl=contour(X,Y,u_k,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(u_new-exacta(X,Y,t[-1]).reshape(N)))
    print ("Error:",error)
    return error
    
def punto_fijo_2(xi,xf,Nx,yi,yf,Ny,ti,tf,M,nu,u0,u1,u2,u3,fun,g,tol):  
    """Resuelve una ecuación del tipo u - nu * \Delta u = f, donde u es una función de
    dos variables, aplicando un esquema de diferencias finitas de segundo orden y construye
    una matriz no simétrica.
    ----------------
    xi:float --- Valor inciial del intervalo en el eje X.
    xf:float --- Valor final del intervalo en el eje X.
    Nx:int --- Número de particiones en el eje X.
    yi:float --- Valor inciial del intervalo en el eje Y.
    yf:float --- Valor final del intervalo en el eje Y.
    Ny:int --- Número de particiones en el eje Y.
    nu:float --- Un número real positivo
    u0:float --- Condición de contorno abajo.
    u1:float --- Condición de contorno a la izquierda.
    u2:float --- Condición de contorno arriba.
    u3:float --- Condición de contorno a la derecha.
    fuente:callable --- Función que define la ecuación.
    ----------------
    """
    iter_max = 500 # Número máximo de iteraciones.
    Nx=int(Nx)
    Ny=int(Ny)
    M = int(M)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)   
    dt = (tf - ti)/int(M)
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    R = lil_matrix((N,N), dtype="float64")
    R.setdiag(ones(N), 0)
    R = R.tocsc()
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    t = linspace(ti,tf,M+1)
    X,Y=meshgrid(x,y) 
    Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx),1)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx),-1)
    My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)
       
    Mx[0,0]=0.0
    Mx[0,1]=0.0
    Mx[Nx,Nx]=0.0
    Mx[Nx,Nx-1]=0.0
        
    My[0,0]=0.0
    My[Nx,Nx]=0.0
    
    for i in range(1,Ny): 
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx   
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My 
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    
    
    A=Id+nu*dt*A
    A=A.tocsc()

    
    g0 = g(X,Y)
    u_k = g0
    u_new = zeros(N)
    
    for k in range(M):
        cont = 0 #Contador de iteraciones.
        error = tol + 1
        u_old = g0
        
        while(error >= tol and cont < iter_max):
            b = dt * fun(X,Y,t[k+1]) + u_k 
            b[0,:]=u0(x,t[k+1])
            b[Ny,:]=u2(x,t[k+1])            
            b[:,0]=u3(y,t[k+1])
            b[:,Nx]=u1(y,t[k+1])
            
            R.setdiag(dt * u_old.reshape(N))
            LU = splu(A + R)
            u_new = LU.solve(b.reshape(N))
            error = max(abs(u_new - u_old.reshape(N)))
            u_old = u_new.reshape((Nx+1,Ny+1))
                
            cont += 1
        
        u_k = u_new.reshape((Nx+1,Ny+1))
            
            
    if(cont == iter_max):
        print("Se ha alcanzado el número máximo de iteraciones:",iter_max)
    else:
        print("Número de iteraciones:",cont)
        
    cu=contourf(X,Y,u_k,20)
    colorbar(cu)
    cl=contour(X,Y,u_k,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(u_new-exacta(X,Y,t[-1]).reshape(N)))
    print ("Error:",error)
    return error

def newton(xi,xf,Nx,yi,yf,Ny,ti,tf,M,nu,u0,u1,u2,u3,fun,g,tol):
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
    Bn = 1e10
    Nx = int(Nx)
    Ny = int(Ny)
    M = int(M)
    xi = float(xi)
    xf = float(xf)
    yi = float(yi)
    yf = float(yf)
    dx = (xf-xi)/float(Nx)
    dy = (yf-yi)/float(Ny)   
    dt = (tf - ti)/int(M)
    N = (Nx+1) * (Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    R = lil_matrix((N,N), dtype="float64")
    R.setdiag(ones(N), 0)
    R = R.tocsc()
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    t = linspace(ti,tf,M+1)
    X,Y=meshgrid(x,y) 
    Mx.setdiag(1 + 2.0 * nu * dt *(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-nu * dt/(dx**2)*ones(Nx),1)
    Mx.setdiag(-nu * dt/(dx**2)*ones(Nx),-1)
    My.setdiag(-nu * dt/(dy**2)*ones(Nx+1),0)
       
    Mx[0,0]=Bn
    Mx[Nx,Nx]=Bn
    
    
    for i in range(1,Ny):
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    Mx.setdiag(Bn*ones(Nx+1),0)
    A[0:(Nx+1),0:(Nx+1)]=Mx
    A[Ny*(Nx+1):(Ny+1)*(Nx+1),Ny*(Nx+1):(Ny+1)*(Nx+1)]=Mx
    
    A=A.tocsc()
    
    g0 = g(X,Y).reshape(N)
    u_new = zeros(N)
    u_n = g0
    
    for n in range(M):
         u_old = g0
         cont = 0 # Contador de iteraciones.
         error = tol + 1
         while(error >= tol and cont < iter_max):
             b = - A * u_old - dt * u_old**2 + dt * fun(X,Y,t[n+1]).reshape(N) + u_n
             b = b.reshape((Nx+1,Ny+1))
             u_old = u_old.reshape((Nx+1,Ny+1))
             u_n = u_n.reshape((Nx+1,Ny+1))
             b[0,:] =  Bn * (u0(x,t[n+1]) - u_old[0,:])
             b[Ny,:] = Bn * (u2(x,t[n+1]) - u_old[Ny,:])           
             b[:,0] = Bn * (u3(y,t[n+1]) - u_old[:,0])
             b[:,Nx] = Bn * (u1(y,t[n+1]) - u_old[:,Nx])
             u_old = u_old.reshape(N)
             u_n = u_n.reshape(N)
             # Contruimos la matriz jacobiana.
             R.setdiag(2 * dt * u_old, 0)
             J = A + R  # El diferencial de la función que describe el método iterativo.
             LU = splu(J)
        
             b = b.reshape(N)
             y_new = LU.solve(b)
             u_new = y_new + u_old
             error = max(abs(u_new - u_old))
             u_old = u_new
        
             cont += 1
         u_n = u_new
        
    if(cont == iter_max):  # Avisamos si se alcanza el número máximo de iteraciones.
        print("Número máximo de iteraciones (", iter_max, ") alcanzado.")
        
        
    cu=contourf(X,Y,u_new.reshape((Nx+1,Ny+1)),20)
    colorbar(cu)
    cl=contour(X,Y,u_new.reshape((Nx+1,Ny+1)),20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(u_new-exacta(X,Y,t[-1]).reshape(N)))
    print ("Error:",error)
    return error
    
################################## PRUEBAS ##################################    
    
    
def f(x,y,t):
    z = cos(x) * sin(y) * (cos(t) + sin(t)**2 * cos(x) * sin(y) + 2 * sin(t))
    return z
def exacta(x,y,t):
    z = cos(x) * sin(y) * sin(t)
    return z
def ui(x,t):
    return 0 
ui = vectorize(ui)
def u3(y,t):
    z = sin(t) * sin(y)
    return z
def g0(x,y):
    return 0 
g0 = vectorize(g0)

# print("Ejercicio 1 \n")

# figure("Ejercicio 1")
# error0 = 1 
# mesh = [5,10,20]
# for N in mesh:
#     error1 = punto_fijo_1(0, 0.5*pi, N, 0, pi, N, 0, 2*pi, 1e4, 1, ui, ui, ui, u3, f, g0, 1e-5)  
#     if N > 5:
#         print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
#     else:
#         print("Con N =", 5, "el orden es = ---- \n")
#     error0 = error1
 
# print("Ejercicio 2 \n")

# figure("Ejercicio 2")
# error0 = 1 
# mesh = [5,10,20]
# for N in mesh:
#     error1 = punto_fijo_2(0, 0.5*pi, N, 0, pi, N, 0, 2*pi, 1e4, 1, ui, ui, ui, u3, f, g0, 1e-5)  
#     if N > 5:
#         print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
#     else:
#         print("Con N =", 5, "el orden es = ---- \n")
#     error0 = error1    
 
print("Ejercicio 3 \n")

figure("Ejercicio 3")
error0 = 1 
mesh = [4,8,16,32]
for N in mesh:
    error1 = newton(0, 0.5*pi, N, 0, pi, N, 0, 2*pi, 1e4, 1, ui, ui, ui, u3, f, g0, 1e-5)  
    if N > 4:
        print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
    else:
        print("Con N =", 5, "el orden es = ---- \n")
    error0 = error1        
 
    
 
    
 
    
 
    
 
    
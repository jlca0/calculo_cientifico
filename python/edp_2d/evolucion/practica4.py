from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix,identity
from matplotlib.pyplot import *
    

def eliptico_2d_dirichlet_penalizacion(xi,xf,Nx,yi,yf,Ny,nu,u0,u1,u2,u3,fuente):
    """Resuelve una ecuación del tipo u - nu * \Delta u = f, donde u es una función de
    dos variables, aplicando un esquema de diferencias finitas de segundo orden y penalización
    en la construcción de la matriz.
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
    Nx=int(Nx)
    Ny=int(Ny)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)
    N=(Nx+1)*(Ny+1)
    Bn=1e+30
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    X,Y=meshgrid(x,y)
    Mx.setdiag(1.0+nu*(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1)),0)
    Mx.setdiag(-nu/(dx**2)*ones(Nx),1)
    Mx.setdiag(-nu/(dx**2)*ones(Nx),-1)
    My.setdiag(-nu/(dy**2)*ones(Nx+1),0)
    
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
    
    b=zeros((Ny+1,Nx+1))
    b=fuente(X,Y)
    
    b[0,:]=Bn*u0(x)
    
    
    b[Ny,:]=Bn*u2(x)
    
    b[:,0]=Bn*u3(y)
    
    
    b[:,Nx]=Bn*u1(y)
    
    
    b=b.reshape(N)
    
    LU=splu(A)
    usol=LU.solve(b)
    
    
    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20)
    colorbar(cu)
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(usol-exacta(X,Y)).reshape(N))
    print ("Error:",error)

def eliptico_2d_dirichlet_simetrizacion(xi,xf,Nx,yi,yf,Ny,nu,u0,u1,u2,u3,fuente):
    """Resuelve una ecuación del tipo u - nu * \Delta u = f, donde u es una función de
    dos variables, aplicando un esquema de diferencias finitas de segundo orden y simetrización
    en la construcción de la matriz.
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
    Nx=int(Nx)
    Ny=int(Ny)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)    
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    X,Y=meshgrid(x,y) 
    Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx),1)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx),-1)
    My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)
       
    Mx[0,0]=0.0
    Mx[0,1]=0.0
    Mx[Nx,Nx]=0.0
    Mx[Nx,Nx-1]=0.0
    
    Mx[1,0]=0.0
    Mx[Nx-1,Nx]=0.0
    
    My[0,0]=0.0
    My[Nx,Nx]=0.0
    
    for i in range(1,Ny): 
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx   
        if (i>1):
            A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My 
        if (i<Ny-1):
            A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    

    A=Id+nu*A
    A=A.tocsc()
   
    b=zeros((Ny+1,Nx+1)) 
    b=fuente(X,Y)
    
    b[0,:]=u0(x)
    b[1,1:Nx]+=nu*u0(x[1:Nx])/(dy*dy)
    
    b[Ny,:]=u2(x)
    b[(Ny-1),1:Nx]+=nu*u2(x[1:Nx])/(dy*dy)
    
    b[:,0]=u3(y)
    b[1:Ny,1]+=nu*u3(y[1:Ny])/(dx*dx)

    b[:,Nx]=u1(y)
    b[1:Ny,(Nx-1)]+=nu*u1(y[1:Ny])/(dx*dx)
    
    b=b.reshape(N) 
    
    LU=splu(A)	    
    usol=LU.solve(b)  
   

    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20)
    colorbar(cu)
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(usol-exacta(X,Y)).reshape(N))
    print ("Error:",error)
    
def eliptico_2d_dirichlet_no_simetrico(xi,xf,Nx,yi,yf,Ny,nu,u0,u1,u2,u3,fuente):  
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
    Nx=int(Nx)
    Ny=int(Ny)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)    
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
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
    
    
    A=Id+nu*A
    A=A.tocsc()
   
    b=zeros((Ny+1,Nx+1)) 
    b=fuente(X,Y)
    
    b[0,:]=u0(x)
    
    b[Ny,:]=u2(x)
    
    b[:,0]=u3(y)

    b[:,Nx]=u1(y)
    
    b=b.reshape(N) 
    
    LU=splu(A)	    
    usol=LU.solve(b)  
   
    tfin=time.time()

    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20)
    colorbar(cu)
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(usol-exacta(X,Y)).reshape(N))
    print ("Error:",error)
    
def eliptico_2d_neumann_derecha_penalizacion(xi,xf,Nx,yi,yf,Ny,nu,u0,un,u2,u3,fuente):
    """Resuelve una ecuación del tipo u - nu * \Delta u = f, donde u es una función de
    dos variables, aplicando un esquema de diferencias finitas de segundo orden y penalización
    en la construcción de la matriz. En la parte inferior del dominio se aplica una condición 
    estilo Neumann.
    ----------------
    xi:float --- Valor inciial del intervalo en el eje X.
    xf:float --- Valor final del intervalo en el eje X.
    Nx:int --- Número de particiones en el eje X.
    yi:float --- Valor inciial del intervalo en el eje Y.
    yf:float --- Valor final del intervalo en el eje Y.
    Ny:int --- Número de particiones en el eje Y.
    nu:float --- Un número real positivo
    u0:float --- Condición Neumann abajo.
    un:float --- Condición de contorno a la derecha.
    u2:float --- Condición de contorno arriba.
    u3:float --- Condición de contorno a la izquierda.
    fuente:callable --- Función que define la ecuación.
    ----------------
    """
    Nx=int(Nx)
    Ny=int(Ny)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)
    N=(Nx+1)*(Ny+1)
    Bn=1e+30
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    X,Y=meshgrid(x,y)
    Mx.setdiag(1.0+nu*(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1)),0)
    Mx.setdiag(-nu/(dx**2)*ones(Nx),1)
    Mx.setdiag(-nu/(dx**2)*ones(Nx),-1)
    My.setdiag(-nu/(dy**2)*ones(Nx+1),0)
    
    Mx[0,0] = Bn
        
    Mx[Nx,Nx-1] = 2 * Mx[Nx,Nx-1]
    
    for i in range(1,Ny):
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    Mx.setdiag(Bn*ones(Nx+1),0)
    A[0:(Nx+1),0:(Nx+1)]=Mx
    A[Ny*(Nx+1):(Ny+1)*(Nx+1),Ny*(Nx+1):(Ny+1)*(Nx+1)]=Mx
    
    A=A.tocsc()
    
    b=zeros((Ny+1,Nx+1))
    b=fuente(X,Y)
    
    b[0,:]=Bn*u0(x)
    
    
    b[Ny,:]=Bn*u2(x)
    
    b[:,0]=Bn*u3(y)
    
    
    b[:,Nx] += 2 * nu / dx * un(y)
    
    
    b=b.reshape(N)
    
    LU=splu(A)
    usol=LU.solve(b)
    
    
    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20)
    colorbar(cu)
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(usol-exacta1(X,Y)).reshape(N))
    print ("Error:",error)
    
def eliptico_2d_neumann_abajo_penalizacion(xi,xf,Nx,yi,yf,Ny,nu,un,u1,u2,u3,fuente):
    """Resuelve una ecuación del tipo u - nu * \Delta u = f, donde u es una función de
    dos variables, aplicando un esquema de diferencias finitas de segundo orden y penalización
    en la construcción de la matriz. En la parte inferior del dominio se aplica una condición 
    estilo Neumann.
    ----------------
    xi:float --- Valor inciial del intervalo en el eje X.
    xf:float --- Valor final del intervalo en el eje X.
    Nx:int --- Número de particiones en el eje X.
    yi:float --- Valor inciial del intervalo en el eje Y.
    yf:float --- Valor final del intervalo en el eje Y.
    Ny:int --- Número de particiones en el eje Y.
    nu:float --- Un número real positivo
    un:float --- Condición Neumann abajo.
    u1:float --- Condición de contorno a la izquierda.
    u2:float --- Condición de contorno arriba.
    u3:float --- Condición de contorno a la derecha.
    fuente:callable --- Función que define la ecuación.
    ----------------
    """
    Nx=int(Nx)
    Ny=int(Ny)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)
    N=(Nx+1)*(Ny+1)
    Bn=1e+30
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    X,Y=meshgrid(x,y)
    Mx.setdiag(1.0+nu*(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1)),0)
    Mx.setdiag(-nu/(dx**2)*ones(Nx),1)
    Mx.setdiag(-nu/(dx**2)*ones(Nx),-1)
    My.setdiag(-nu/(dy**2)*ones(Nx+1),0)
    
    A[0:(Nx+1),0:(Nx+1)]=Mx  # No modificamos la primera fila de matrices.
    A[0:(Nx+1),(Nx+1):2*(Nx+1)]= 2 * My
    
    # Condiciones de contorno en gamma1, gamma3
    Mx[0,0]=Bn
    Mx[Nx,Nx]=Bn
    
    for i in range(1,Ny):
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    Mx.setdiag(Bn*ones(Nx+1),0)
    A[Ny*(Nx+1):(Ny+1)*(Nx+1),Ny*(Nx+1):(Ny+1)*(Nx+1)]=Mx
    
    A=A.tocsc()
    
    b=zeros((Ny+1,Nx+1))
    b=fuente(X,Y)
    
    
    
    b[Ny,:]=Bn*u2(x)
    
    b[:,0]=Bn*u3(y)
    
    b[:,Nx] = Bn*u1(y)
    
    b[0,:] -= 2* nu/dy * un(x)
    
    
    b=b.reshape(N)
    
    LU=splu(A)
    usol=LU.solve(b)
    
    
    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20)
    colorbar(cu)
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(usol-exacta2(X,Y)).reshape(N))
    print ("Error:",error)

def eliptico_2d_evolucion_implicito(xi,xf,Nx,yi,yf,Ny,ti,tf,M,nu,u0,un,u2,u3,fuente,g):
    """Resuelve una ecuación del tipo u - nu * \Delta u = f, donde u es una función de
    dos variables, aplicando un esquema de diferencias finitas de segundo orden y penalización
    en la construcción de la matriz.
    ----------------
    xi:float --- Valor inciial del intervalo en el eje X.
    xf:float --- Valor final del intervalo en el eje X.
    Nx:int --- Número de particiones en el eje X.
    yi:float --- Valor inciial del intervalo en el eje Y.
    yf:float --- Valor final del intervalo en el eje Y.
    Ny:int --- Número de particiones en el eje Y.
    nu:float --- Un número real positivo
    u0:float --- Condición de contorno abajo.
    un:float --- Condición de contorno a la derecha.
    u2:float --- Condición de contorno arriba.
    u3:float --- Condición de contorno a la izquierda.
    fuente:callable --- Función que define la ecuación.
    ----------------
    """
    Nx=int(Nx)
    Ny=int(Ny)
    M=int(M)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    ti=float(ti)
    tf=float(tf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)
    dt = (tf - ti)/float(M)
    N=(Nx+1)*(Ny+1)
    Bn=1e+30
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    t = linspace(ti, tf, M+1)
    X,Y=meshgrid(x,y)
    Mx.setdiag(1.0+nu * dt * (2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1)),0)
    Mx.setdiag(-nu * dt/(dx**2)*ones(Nx),1)
    Mx.setdiag(-nu * dt/(dx**2)*ones(Nx),-1)
    My.setdiag(-nu * dt/(dy**2)*ones(Nx+1),0)
    
    Mx[0,0] = Bn
        
    Mx[Nx,Nx-1] = 2 * Mx[Nx,Nx-1]
    
    for i in range(1,Ny):
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    Mx.setdiag(Bn*ones(Nx+1),0)
    A[0:(Nx+1),0:(Nx+1)]=Mx
    A[Ny*(Nx+1):(Ny+1)*(Nx+1),Ny*(Nx+1):(Ny+1)*(Nx+1)]=Mx
    
    A=A.tocsc()   
    LU=splu(A)
    
    usol = g(X, Y)
    
    for k in range(M):
        b = dt * fuente(X,Y,t[k+1]) + usol
        
         
        b[0,:]=Bn*u0(x,t[k+1])
        b[Ny,:]=Bn*u2(x,t[k+1])        
        b[:,0]=Bn*u3(y,t[k+1])   
        b[1:Ny,Nx] +=  2 * dt * nu / dx * un(y,t[k+1])           
        
        
        b=b.reshape(N)
        usol = usol.reshape(N)
        
        usol = LU.solve(b)
        
        b = b.reshape((Ny+1,Nx+1))
        usol = usol.reshape((Ny+1,Nx+1))
        
    cu=contourf(X,Y,usol,20)
    colorbar(cu)
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(usol-exacta4(X,Y,t[-1])).reshape(N))
    print ("Error:",error)
    return error

def eliptico_2d_theta_metodo(xi,xf,Nx,yi,yf,Ny,ti,tf,M,nu,u0,un,u2,u3,fuente,g,theta):
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
    Nx=int(Nx)
    Ny=int(Ny)
    M = int(M)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    ti=float(ti)
    tf=float(tf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)
    dt = (tf-ti)/float(M)    
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
    Mx[Nx,Nx-1]= 2 * Mx[Nx,Nx-1]
        
    My[0,0]=0.0
    
    for i in range(1,Ny): 
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx   
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My 
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
      
    AE = Id - (1-theta) * nu * dt * A
    AI = Id + theta * nu * dt * A
    AE = AE.tocsc()
    AI = AI.tocsc()
    
    LU = splu(AI)
    usol = g(X, Y)
    
    for k in range(M):
        usol = usol.reshape(N)
        b = dt * (theta * fuente(X,Y,t[k+1]).reshape(N) + (1-theta) * fuente(X,Y,t[k]).reshape(N)) + AE * usol
        b = b.reshape((Nx+1,Ny+1))
        b[0,:]=u0(x,t[k+1])        
        b[Ny,:]=u2(x,t[k+1])        
        b[:,0]=u3(y,t[k+1])
        b[1:Ny,Nx] +=  2 * nu * dt/dx * (theta * un(y,t[k+1]) + (1 - theta) * un(y,t[k]))
        b = b.reshape(N)
        
        usol =  LU.solve(b) 
        
        usol = usol.reshape((Ny+1,Nx+1)) 
        usol[0,:]=u0(x,t[k+1])
        usol[Ny,:]=u2(x,t[k+1])        
        usol[:,0]=u3(y,t[k+1])   
        #usol[1:Ny,Nx] +=  2 * dt * nu / dx * un(y,t[k+1])
        

    cu=contourf(X,Y,usol,20)
    colorbar(cu)
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(usol-exacta4(X,Y,t[-1])).reshape(N))
    print ("Error:",error)
    return error

def eliptico_2d_evolucion_explicito(xi,xf,Nx,yi,yf,Ny,ti,tf,M,nu,u0,un,u2,u3,fuente,g):
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
    Nx=int(Nx)
    Ny=int(Ny)
    M = int(M)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    ti=float(ti)
    tf=float(tf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)
    dt = (tf-ti)/float(M)    
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
    Mx[Nx,Nx-1]= 2 * Mx[Nx,Nx-1]
        
    My[0,0]=0.0
    
    for i in range(1,Ny): 
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx   
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My 
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    
    
    A = Id - nu * dt * A
    A = A.tocsc()
    
    usol = g(X, Y)
    
    for k in range(M):
        usol = usol.reshape(N)
        b = dt * fuente(X,Y,t[k+1])
        b[0,:]=u0(x,t[k+1])        
        b[Ny,:]=u2(x,t[k+1])        
        b[:,0]=u3(y,t[k+1])
        b[1:Ny,Nx] +=  2 * dt * nu / dx * un(y[1:Ny],t[k+1])
        
        usol = A * usol + b.reshape(N) 
        
        usol = usol.reshape((Ny+1,Nx+1)) 
        usol[0,:]=u0(x,t[k+1])
        usol[Ny,:]=u2(x,t[k+1])        
        usol[:,0]=u3(y,t[k+1])   
        #usol[1:Ny,Nx] +=  2 * dt * nu / dx * un(y,t[k+1]) 

    cu=contourf(X,Y,usol,20)
    colorbar(cu)
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fontsize=8)
    show()
    error=max(abs(usol-exacta4(X,Y,t[-1])).reshape(N))
    print ("Error:",error)
    return error

############ PRUEBAS ############


# def f0(x,y):
#     """Función que define el problema de contorno."""
#     z=sin(x*y)*(1+x**2+y**2)
#     return z
# def u0(x):
#     z=0*x 
#     return z
# def u1(y):
#     z=sin(2*pi*y) 
#     return z
# def u2(x):
#     z=sin(2*pi*x) 
#     return z
# def u3(y):
#     z=0*y 
#     return z
# def exacta(x,y):
#     """Solución exacta del problema de contorno."""
#     z=sin(x*y)
#     return z
  
def f1(x,y):
    """Función que define el problema de contorno."""
    z = 2.0 * sin(x)
    return z 
def u01(x):
    z = sin(x)
    return z
def u11(y): 
    return -1
def u31(y):
    return 0
def exacta1(x,y):
    """Solución exacta del problema de contorno."""
    z = sin(x)
    return z

def f2(x,y):
    """Función que define el problema de contorno."""
    z = 2.0 * sin(y)
    return z 
def u02(x):
    return 1
def exacta2(x,y):
    """Solución exacta del problema de contorno."""
    z = sin(y)
    return z

def f2(x,y):
    """Función que define el problema de contorno."""
    z = 2.0 * sin(y)
    return z 
def u02(x):
    return 1
def exacta2(x,y):
    """Solución exacta del problema de contorno."""
    z = sin(y)
    return z

# def f4(x,y,t):
#     """Función que define el problema de contorno."""
#     z = 2.0 * sin(y)
#     return z 
# def u04(x,t):
#     return 1
# def u14(x,t):
#     z = sin(x)
#     return z
# def u34(y,t):
#     return 0
# def exacta4(x,y,t):
#     """Solución exacta del problema de contorno."""
#     z = sin(y)
#     return z

def f4(x,y,t):
    """Función en dos variables que determina la ecuación del calor."""
    z = x * cos(x * t) + t**2 * sin(x * t)
    return z
def exacta4(x,y,t):
    """Solución exacta de la ecuación del calor."""
    z = sin(x * t)
    return z
def u04(x,t):
    z = sin(x * t)
    return z
def u14(y,t):
    z = t * cos(t)  
    return z
def u24(x,t):
    z = sin(x * t)
    return z
def u34(y,t):
    return 0
def g4(x,y):
    return  0
g4 = vectorize(g4)


# figure("Ejercicio 1")
# eliptico_2d_dirichlet_penalizacion(0.0,2*pi,200,0.0,2*pi,200,1.0,u0,u1,u2,u3,f0)

# figure("Ejercicio 2")
# print("Ejercicio 2")
# eliptico_2d_neumann_derecha_penalizacion(0.0,pi,200,0.0,pi,200,1.0,u01,u11,u01,u31,f1)

# figure("Ejercicio 3")
# print("\n")
# print("Ejercicio 3")
# eliptico_2d_neumann_abajo_penalizacion(0.0,pi,200,0.0,pi,200,1.0,u02,u01,u31,u01,f2)

# figure("Ejercicio 4.a.")
# print("\n")
# print("Evolucion implícito. Orden en espacio.")
# mesh = [5,10,20,40]
# error0 = 1
# for N in mesh:
#     error1 = eliptico_2d_evolucion_implicito(0,1,N,0,1,N,0,1,1e4,1,u04,u14,u24,u34,f4,g4)

#     if N > 5:
#         print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
#     else:
#         print("Con N =", 5, "el orden es = ---- \n")
#     error0 = error1
# print("Evolucion implícito. Orden en tiempo.")
# mesh = [5,10,20,40]
# error0 = 1
# for N in mesh:
#     error1 = eliptico_2d_evolucion_implicito(0,1,1e2,0,1,1e2,0,1,N,1,u04,u14,u24,u34,f4,g4)

#     if N > 5:
#         print("Con N =", N, "el orden es =", log(error0 / error1)/log(2), "\n")
#     else:
#         print("Con N =", 5, "el orden es = ---- \n")
#     error0 = error1
    
figure("Ejercicio 4.c.")
eliptico_2d_evolucion_explicito(0,1,100,0,1,100,0,1,5e4,1,u04,u14,u24,u34,f4,g4)

figure("Ejercico 4.c.2")
eliptico_2d_theta_metodo(0,1,100,0,1,100,0,1,5e4,1,u04,u14,u24,u34,f4,g4,0.5)
#eliptico_2d_dirichlet_simetrizacion(0.0,2*pi,200,0.0,2*pi,200,1.0,u0,u1,u2,u3,f0)
#eliptico_2d_dirichlet_no_simetrico(0.0,2*pi,200,0.0,2*pi,200,1.0,u0,u1,u2,u3,f0)

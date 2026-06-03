import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix, identity

def coordenadas_totales(elemento: int, nodo: int) -> int:
    """
    Dado un nodo de un elemento, ordenados ascendentemente de izquierda a derecha,
    da su coordenada total en la partición. Los elementos son ordenados como I_1, ..., I_{N+1}.
    """
    if nodo == 0:
        return elemento-1
    else:
        return elemento

def pol_lagrange(elemento: int, nodo: int, malla: np.ndarray, x: np.ndarray)->np.ndarray:
    """
    Polinomio de base de Lagrange, vale 1 en el nodo del elemento seleccionado.
    """
    i = coordenadas_totales(elemento, nodo)
    if nodo == 0:   
        lmbd= (x - malla[i+1])/(malla[i]-malla[i+1])
    else:
        lmbd= (x - malla[i-1])/(malla[i]-malla[i-1])
    return lmbd

def dpol_lagrange(elemento: int, nodo: int, malla: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Derivada del polinomio de base de Lagrange.
    """
    i = coordenadas_totales(elemento, nodo)
    if nodo == 0:   
        lmbd= 1/(malla[i]-malla[i+1])
    else:
        lmbd= 1/(malla[i]-malla[i-1])
    return lmbd

def simpson(fun: callable, a: float, b: float)-> float:
    """
    Dada una función en un intervalo aproxima su integral por la
    regla de Simpson, que tiene grado de precisión 2.
    """
    cuad = (b-a)/6 * (fun(a) + 4 * fun(0.5*(a+b)) + fun(b))
    return cuad

def rellenar_termino_ind(fuente: callable, malla: np.ndarray) -> np.ndarray:
    """
    Calcula el término independiente iterando sobre los elementos.
    """
    N = len(malla) - 2
    termino_ind = np.zeros(N)

    i0 = coordenadas_totales(1, 1)
    def fun_lmbd(x, nodo=1, elemento=1):
                return fuente(x) * pol_lagrange(elemento, nodo, malla, x)
    termino_ind[i0-1] += simpson(fun_lmbd, malla[0], malla[1]) 

    for elemento in range(2, N+1):
        a, b = malla[elemento - 1], malla[elemento]

        for nodo in range(2):
            i = coordenadas_totales(elemento, nodo)
            
            def fun_lmbd(x, nodo=nodo, elemento=elemento):
                return fuente(x) * pol_lagrange(elemento, nodo, malla, x)
                
            termino_ind[i-1] += simpson(fun_lmbd, a, b)     
    
    iN = coordenadas_totales(N+1, 0)
    def fun_lmbd(x, nodo=0, elemento=N+1):
        return fuente(x) * pol_lagrange(elemento, nodo, malla, x)
    termino_ind[iN-1] += simpson(fun_lmbd, malla[N], malla[N+1])
            
    return termino_ind 

def rellenar_matrices(malla):
    """Calcula las matrices de rigidez y de masa para P1, usando la regla de Simpson 
    e iterando elemento a elemento"""
    N = len(malla) - 2
    R = lil_matrix((N, N))
    M = lil_matrix((N, N))

    for elemento in range(1, N+2):
        a, b = malla[elemento-1], malla[elemento]
        
        # Matrices locales
        R_loc = np.zeros((2,2))
        M_loc = np.zeros((2,2))
        for j in range(2):
            for k in range(2):
                def fun_masa(x, j=j, k=k):
                    return pol_lagrange(elemento, j, malla, x) * pol_lagrange(elemento, k, malla, x)
                def fun_rigidez(x, j=j, k=k):
                    return dpol_lagrange(elemento, j, malla, x) * dpol_lagrange(elemento, k, malla, x)
                R_loc[j,k] = simpson(fun_rigidez, a, b)
                M_loc[j,k] = simpson(fun_masa, a, b)

                ig = coordenadas_totales(elemento, j)
                ik = coordenadas_totales(elemento, k)
                if 1 <= ig <= N and 1 <= ik <= N:
                    R[ig-1, ik-1] += R_loc[j,k]
                    M[ig-1, ik-1] += M_loc[j,k]

    return R.tocsc(), M.tocsc()

def mef(malla: np.ndarray, q: float, fun:callable) -> tuple:
    """
    Resuelve la ecuación -u'' + qu = f mediante el método de elementos finitos
    asociado al espacio P1 en dimensión 1.
    """
    R, M = rellenar_matrices(malla)
    termino_ind = rellenar_termino_ind(fun, malla)
    LU = splu(R + q * M)
    return LU.solve(termino_ind)


###### EJEMPLO ######

N = 20
malla20 = np.linspace(0,1,N)
malla40 = np.linspace(0,1,2*N)
malla80 = np.linspace(0,1,4*N)
malla160 = np.linspace(0,1,8*N)
mallas = [malla20, malla40, malla80, malla160]

def fun(x):
    return (4*np.pi**2 + 1) * np.sin(2*np.pi*x)

def exacta(x):
    return np.sin(2*np.pi*x)

error0 = 1
for malla in mallas:
    sol = mef(malla, 1, fun)
    error1 = max(np.abs(sol - exacta(malla[1:-1])))
    print(f'El orden es = {np.log(error0/error1)/np.log(2):.4f}')
    error0 = error1

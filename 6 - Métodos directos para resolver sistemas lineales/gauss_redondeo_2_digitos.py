import numpy as np
import re

# Redondeo numero n a k digitos
def rnd(n, k):
    if n == 0: return 0
    s = re.search(r'.*e([+|-]){1}([0-9]*)', 
                  format(n, "."+ str(k) + "e"))    
    if s.group(1) == "-": e = -1*int(s.group(2))
    else : e = int(s.group(2))
    n = n + 5*10**(e - (k+1))
    return float(format(n, "."+ str(k-1) + "e"))

def rnd_v(v, k):
    return np.array([rnd(j, k) for j in v])

def g_back_subs(A, b):
    n = len(A)

    E = np.zeros([n, n+1], dtype=float)    
    for i in range(n): 
        E[i, :-1] = A[i, :]
        E[i, n] = b[i]

    x = np.zeros(n)

    for i in range(n):        
        T = [j for j in E[i:,i] if j !=0]
        if len(T) == 0:
            print("No unique solution")
            return 
        p = i + list(E[i:,i]).index(T[0])            
        if p != i:
            print("E", p, "<-> E", i)
            E[[i, p]] = E[[p, i]]
        for j in range(i+1, n):
            m = rnd(E[j, i]/E[i, i], 2)
            print("m=", m)
            temp = rnd_v(m*E[i, :], 2)
            E[j, :] = rnd_v(E[j, :] - temp, 2)
            print(E, "\n")

    if E[n-1, n-1] == 0:
        print("No solution exists")
        return    
    x[n-1] = rnd(E[n-1, n]/E[n-1, n-1], 2)
    for i in reversed(range(n-1)):        
        temp = [rnd(E[i, j]*x[j], 2) for j in range(i, n)]
        temp = rnd(sum(temp), 2)
        temp = rnd(E[i, n] - temp, 2)
        x[i] = rnd(temp/E[i, i],2)
    print(x)

# problem (a)
A = np.array([[4, -1, 1], [2, 5, 2], [1, 2, 4]], dtype=float)
b = np.array([[8], [3], [11]], dtype = float)
g_back_subs(A, b)

# problem (b)
A = np.array([[4, 1, 2], [2, 4, -1], [1, 1, -3]], dtype=float)
b = np.array([[9], [-5], [-9]], dtype = float)
g_back_subs(A, b)
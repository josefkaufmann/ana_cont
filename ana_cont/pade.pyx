"""
Very inefficient implementation of Pade approximation,
described in H. J. Vidberg and J. W. Serene, J. Low Temp. Phys. 29, 179 (1977).

Author: Josef Kaufmann
"""

import numpy as np
import sys


def compute_coefficients(zi,ui):
  cdef int i,j,n
  n=len(zi)
  g=np.zeros((n,n),dtype=complex)
  g[:,0]=ui
  for i in range(n):
    for j in range(1,i+1):
      g[i,j]=(g[j-1,j-1]-g[i,j-1])/((zi[i]-zi[j-1])*g[i,j-1])
  return np.diag(g)


def a(zi,ui):
  cdef int n
  n=len(zi)
  alist=np.zeros((n-1),dtype=np.double)
  for i in range(1,n):
    alist[i-1]=g(i,i,zi,ui)
  return alist

def g(int p,int i,zi,ui):
  if p==1:
    return ui[i-1]
  elif p>1:
    return (g(p-1,p-1,zi,ui)-g(p-1,i,zi,ui))/((zi[i-1]-zi[p-2])*g(p-1,i,zi,ui))

def A(zr,int n,zi,ui,a):
  if n==0:
    return 0.
  elif n==1:
    return a[0]
  else:
    return A(zr,n-1,zi,ui,a) + (zr-zi[n-2])*a[n-1]*A(zr,n-2,zi,ui,a)

def B(zr,int n,zi,ui,a):
  if n==0:
    return 1.
  elif n==1:
    return 1.
  else:
    return B(zr,n-1,zi,ui,a) + (zr-zi[n-2])*a[n-1]*B(zr,n-2,zi,ui,a)

def C(zr,zi,ui,a):
  cdef int n
  n=len(zi)
  return A(zr,n,zi,ui,a)/B(zr,n,zi,ui,a)



import numpy as np
import matplotlib.pyplot as plt




def gauss_peak(maxpos,width,weight,wgrid):
    a =weight/(np.sqrt(2.*np.pi)*width)*np.exp(-0.5*(wgrid-maxpos)**2/width**2)
    return a

def noise(sigma,iwgrid):
    return np.random.randn(iwgrid.shape[0])*sigma

nw=1001
ntau=1000
beta=1.
wgrid=np.linspace(-5/beta,5/beta,nw)
taugrid=np.linspace(0.,beta,num=ntau,endpoint=True)
chi_tau=np.zeros_like(taugrid,dtype=np.complex)
sigma=0.0001

aw=gauss_peak(-1,0.2,1.,wgrid)+gauss_peak(1.,0.2,1.,wgrid)
norm=np.trapz(aw,wgrid)
aw=aw/norm

np.savetxt('spectrum.dat',np.vstack((wgrid,aw)).transpose())

for i,tau in enumerate(taugrid):
  chi_tau[i]=np.trapz(aw*np.exp(-tau*wgrid)/(1.+np.exp(-beta*wgrid)),wgrid)

chi_tau+=noise(sigma,taugrid)

plt.figure()
plt.plot(taugrid,chi_tau.real)
plt.show()

err=np.ones_like(taugrid)*sigma

np.savetxt('test.dat',np.vstack((taugrid,chi_tau.real,err)).transpose())

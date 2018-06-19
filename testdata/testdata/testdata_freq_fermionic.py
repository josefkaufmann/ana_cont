import numpy as np
import matplotlib.pyplot as plt




def gauss_peak(maxpos,width,weight,wgrid):
    a =weight/(np.sqrt(2.*np.pi)*width)*np.exp(-0.5*(wgrid-maxpos)**2/width**2)
    return a

def noise(sigma,iwgrid):
    return np.random.randn(iwgrid.shape[0])*sigma

nw=1001
niw=100
beta=10.
wgrid=np.linspace(-5,5,nw)
iwgrid=np.pi/beta*(2*np.arange(niw)+1)
giw=np.zeros_like(iwgrid,dtype=np.complex)
sigma=0.0001

aw=gauss_peak(-1.,0.2,1.,wgrid)+gauss_peak(1.,0.2,1.,wgrid)
norm=np.trapz(aw,wgrid)
aw=aw/norm

np.savetxt('spectrum.dat',np.vstack((wgrid,aw)).transpose())

for i,iw in enumerate(iwgrid):
  giw[i]=-np.trapz(aw*wgrid/(iw**2+wgrid**2),wgrid)-1j*np.trapz(aw*iw/(iw**2+wgrid**2),wgrid)

giw+=noise(sigma,iwgrid)

plt.figure()
plt.plot(iwgrid,giw.real)
plt.plot(iwgrid,giw.imag)
plt.show()

err=np.ones_like(iwgrid)*sigma

np.savetxt('giw.dat',np.vstack((iwgrid,giw.real,giw.imag,err)).transpose())

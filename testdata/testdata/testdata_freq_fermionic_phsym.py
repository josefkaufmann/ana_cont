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
wgrid=np.linspace(0,5,nw)
iwgrid=np.pi/beta*(2*np.arange(niw)+1)
giw=np.zeros_like(iwgrid,dtype=np.complex)
sigma=0.01

xm=1.
wid=0.4
aw=gauss_peak(xm,wid,1.,wgrid)+gauss_peak(-xm,wid,1.,wgrid) # one gauss peak alone is not ph symmetric
norm=np.trapz(aw,wgrid)
aw=aw/norm

np.savetxt('spectrum.dat',np.vstack((wgrid,aw)).transpose())

for i,iw in enumerate(iwgrid):
  giw[i]=-np.trapz(aw*2.*iw/(iw**2+wgrid**2),wgrid)

giw+=noise(sigma,iwgrid)

plt.figure()
plt.plot(iwgrid,giw.real)
plt.show()

err=np.ones_like(iwgrid)*sigma

np.savetxt('giw.dat',np.vstack((iwgrid,giw.real,err)).transpose())

import numpy as np
import matplotlib.pyplot as plt




def gauss_peak(maxpos,width,weight,wgrid):
    a =weight/(np.sqrt(2.*np.pi)*width)*np.exp(-0.5*(wgrid-maxpos)**2/width**2)
    a-=weight/(np.sqrt(2.*np.pi)*width)*np.exp(-0.5*(wgrid+maxpos)**2/width**2)
    return a

def noise(sigma,iwgrid):
    return np.random.randn(iwgrid.shape[0])*sigma


wgrid=np.linspace(0.,15.,1001)
beta=50.
iwgrid=2*np.pi/beta*np.arange(250)
chi_iw=np.zeros_like(iwgrid,dtype=np.complex)
sigma=0.0001

aw=gauss_peak(4.,0.2,1.,wgrid)+gauss_peak(6.,0.5,1.,wgrid)+gauss_peak(9,0.5,1.,wgrid)

norm=np.trapz(aw[1:]/wgrid[1:],wgrid[1:])*2./np.pi
aw=aw/norm


np.savetxt('spectrum.dat',np.vstack((wgrid,aw)).transpose())

for i,iw in enumerate(iwgrid):
  chi_iw[i]=np.trapz(aw*wgrid/(wgrid**2+iw**2),wgrid)
chi_iw[0]=np.trapz(aw[1:]/wgrid[1:],wgrid[1:])
chi_iw=chi_iw*2/np.pi

chi_iw+=noise(sigma,iwgrid)

#plt.figure()
#plt.plot(iwgrid,chi_iw.real)
#plt.show()

err=np.ones_like(iwgrid)*sigma

np.savetxt('test.dat',np.vstack((iwgrid,chi_iw.real,err)).transpose())

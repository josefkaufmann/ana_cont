import numpy as np
import matplotlib.pyplot as plt




def gauss_peak(maxpos,width,weight,wgrid):
    a =weight/(np.sqrt(2.*np.pi)*width)*np.exp(-0.5*(wgrid-maxpos)**2/width**2)
    a-=weight/(np.sqrt(2.*np.pi)*width)*np.exp(-0.5*(wgrid+maxpos)**2/width**2)
    return a

def noise(sigma,iwgrid):
    return np.random.randn(iwgrid.shape[0])*sigma


wgrid=np.linspace(0.,15.,1001)
beta=5.
iwgrid=2*np.pi/beta*np.arange(20)
taugrid=np.linspace(0.,beta,num=501,endpoint=True)
chi_tau=np.zeros_like(taugrid,dtype=np.float)
sigma=1.

aw=gauss_peak(2.,0.2,1.,wgrid)#+gauss_peak(6.,0.5,1.,wgrid)+gauss_peak(9,0.5,1.,wgrid)

norm=np.trapz(aw[1:]/wgrid[1:],wgrid[1:])
aw=aw/norm


np.savetxt('spectrum.dat',np.vstack((wgrid,aw)).transpose())

for i,tau in enumerate(taugrid):
    integrand=aw*0.5*(np.exp(-wgrid*tau)+np.exp(-wgrid*(beta-tau)))/(1.-np.exp(-wgrid*beta))
    integrand[0]=0.
    chi_tau[i]=np.trapz(integrand,wgrid)

chi_iw=np.zeros_like(iwgrid,dtype=np.complex)
for i,iw in enumerate(iwgrid):
    chi_iw[i]=np.trapz(chi_tau*np.exp(1j*taugrid*iw),taugrid)

chi_iw_direct=np.zeros_like(chi_iw)
for i,iw in enumerate(iwgrid):
    chi_iw_direct[i]=np.trapz(aw*wgrid/(wgrid**2+iw**2),wgrid)
chi_iw_direct[0]=np.trapz(aw[1:]/wgrid[1:],wgrid[1:])
#chi_iw_direct=chi_iw_direct*2/np.pi

plt.figure()
plt.plot(iwgrid,chi_iw.real)
plt.plot(iwgrid,chi_iw_direct.real)
plt.show()

#chi_tau+=noise(sigma,taugrid)
plt.figure()
plt.plot(taugrid,chi_tau)
plt.show()

err=np.ones_like(taugrid)*sigma

np.savetxt('test.dat',np.vstack((taugrid,chi_tau,err)).transpose())

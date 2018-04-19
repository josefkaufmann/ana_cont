#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ana_cont



iwgrid,giw_re,giw_im,err=np.loadtxt('testdata/giw.dat').transpose()
truewgrid,truespec=np.loadtxt('testdata/spectrum.dat').transpose()


# create a grid that is denser in the center and more sparse outside.
wmax=5.
wcenter=0.
nw=250
wgrid=np.tan(np.linspace(-np.pi/2.5,np.pi/2.5,num=2*nw+1,endpoint=True))*wmax/np.tan(np.pi/2.5)+wcenter

#wgrid=np.linspace(-5,5.,501,endpoint=True) # or just use an ordinary linspace
#model=np.ones_like(wgrid)
model=np.exp(-(wgrid)**2)
model=model/np.trapz(model,wgrid) # normalize the model
niw=iwgrid.shape[0]


# optionally rescale the error
err*=1.


probl=ana_cont.AnalyticContinuationProblem(im_axis=iwgrid,re_axis=wgrid,im_data=giw_re+1j*giw_im,kernel_mode='freq_fermionic')

sol=probl.solve(method='maxent_svd',model=model,stdev=err,alpha_determination='classic')

f1=plt.figure()
p1=f1.add_subplot(131)
p2=f1.add_subplot(132)
p3=f1.add_subplot(133)
p1.plot(wgrid,sol[0].A_opt)
p1.plot(truewgrid,truespec)
p2.plot(iwgrid,sol[0].backtransform.real)
p2.plot(iwgrid,sol[0].backtransform.imag)
p2.errorbar(iwgrid,giw_re,yerr=err)
p2.errorbar(iwgrid,giw_im,yerr=err)
p3.plot(iwgrid,sol[0].backtransform.real-giw_re)
p3.plot(iwgrid,sol[0].backtransform.imag-giw_im)
f1.show()

raw_input()

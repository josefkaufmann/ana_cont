#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ana_cont



iwgrid,giw_re,giw_im,err=np.loadtxt('testdata/giw.dat').transpose()
truewgrid,truespec=np.loadtxt('testdata/spectrum.dat').transpose()
wgrid=np.linspace(-5,5.,500)
#model=np.ones_like(wgrid)
model=np.exp(-(wgrid)**2)
model=model/np.trapz(model,wgrid)
niw=iwgrid.shape[0]

probl=ana_cont.AnalyticContinuationProblem(im_axis=iwgrid,re_axis=wgrid,im_data=1j*giw_im,kernel_mode='freq_fermionic')

sol=probl.solve(method='maxent_svd',model=model,stdev=1*err,alpha_determination='classic')

f1=plt.figure()
p1=f1.add_subplot(131)
p2=f1.add_subplot(132)
p3=f1.add_subplot(133)
p1.plot(wgrid,sol[0].A_opt)
p1.plot(truewgrid,truespec)
p2.plot(iwgrid,sol[0].backtransform.real)
p2.plot(iwgrid,sol[0].backtransform.imag)
p2.plot(iwgrid,giw_re,marker='x',linestyle='None')
p2.plot(iwgrid,giw_im,marker='+',linestyle='None')
p3.plot(iwgrid,sol[0].backtransform.real-giw_re)
p3.plot(iwgrid,sol[0].backtransform.imag-giw_im)
f1.show()

raw_input()

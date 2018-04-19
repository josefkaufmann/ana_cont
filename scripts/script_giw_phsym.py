#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ana_cont



iwgrid,giw,err=np.loadtxt('testdata/giw.dat').transpose()
truewgrid,truespec=np.loadtxt('testdata/spectrum.dat').transpose()
wgrid=(np.exp(np.linspace(0.,5.,500))-1.)*5./(np.exp(5.)-1.)
#model=np.ones_like(wgrid)
model=wgrid**2*np.exp(-(wgrid))
model=model/np.trapz(model,wgrid)
niw=iwgrid.shape[0]

probl=ana_cont.AnalyticContinuationProblem(im_axis=iwgrid,re_axis=wgrid,im_data=giw,kernel_mode='freq_fermionic_phsym')

sol=probl.solve(method='maxent_svd',model=model,stdev=10*err)

f1=plt.figure(1)
p1=f1.add_subplot(121)
p2=f1.add_subplot(122)
p1.plot(wgrid,sol[0].A_opt)
p1.plot(truewgrid,truespec)
p2.plot(iwgrid,sol[0].backtransform)
p2.plot(iwgrid,giw,marker='x',linestyle='None')
f1.show()

raw_input()

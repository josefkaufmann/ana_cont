#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ana_cont



taugrid,gtau,err=np.loadtxt('testdata/gtau.dat').transpose()
truewgrid,truespec=np.loadtxt('testdata/spectrum.dat').transpose()
wgrid=np.linspace(-5,5.,500)
model=np.ones_like(wgrid)
model=model/np.trapz(model,wgrid)

# IMPORTANT: if a time kernel is used, the user has to specify beta!
probl=ana_cont.AnalyticContinuationProblem(im_axis=taugrid,re_axis=wgrid,im_data=gtau,kernel_mode='time_fermionic',beta=1.)

sol=probl.solve(method='maxent_svd',model=model,stdev=10*err)

f1=plt.figure(1)
p1=f1.add_subplot(121)
p2=f1.add_subplot(122)
p1.plot(wgrid,sol[0].A_opt)
p1.plot(truewgrid,truespec)
p2.plot(taugrid,sol[0].backtransform*np.pi/2.)
p2.plot(taugrid,gtau*np.pi/2.,marker='.',linestyle='None')
f1.show()

raw_input()

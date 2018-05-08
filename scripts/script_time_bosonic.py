#!/usr/bin/env python2.7
import numpy as np
import matplotlib.pyplot as plt
import ana_cont



iwgrid,giw,err=np.loadtxt('testdata/test.dat').transpose()
truewgrid,truespec=np.loadtxt('testdata/spectrum.dat').transpose()
wgrid=np.linspace(0.,15.,500)
model=np.ones_like(wgrid)
model=model/np.trapz(model,wgrid)

probl=ana_cont.AnalyticContinuationProblem(im_axis=iwgrid,re_axis=wgrid,im_data=giw,kernel_mode='time_bosonic',beta=5.)

sol=probl.solve(method='maxent_svd',model=model,stdev=0.01*err,alpha_determination='classic')

f1=plt.figure(1)
p1=f1.add_subplot(121)
p2=f1.add_subplot(122)
p1.plot(wgrid,wgrid*sol[0].A_opt)
p1.plot(truewgrid,truespec)
p2.plot(iwgrid,sol[0].backtransform)
p2.plot(iwgrid,giw,marker='.',linestyle='None')

f1.show()
raw_input()

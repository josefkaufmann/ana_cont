#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ana_cont.ana_cont.ana_cont

# NOTE: the kernel 'time_fermionic_phsym' does not work well for insulating systems
# in any case it is not worse to use the usual 'time_fermionic' kernel
# and make the data ph-symmetric


taugrid,gtau,err=np.loadtxt('testdata/gtau.dat').transpose()
truewgrid,truespec=np.loadtxt('testdata/spectrum.dat').transpose()
wgrid=np.linspace(0.,5.,500)
model=np.ones_like(wgrid)
model=model/np.trapz(model,wgrid)

probl= ana_cont.ana_cont.AnalyticContinuationProblem(im_axis=taugrid, re_axis=wgrid, im_data=gtau, kernel_mode='time_fermionic_phsym')

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

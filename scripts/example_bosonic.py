import sys, os
import numpy as np
import matplotlib.pyplot as plt

file_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, package_dir)  # for applications, replace package_dir by the location of ana_cont
import ana_cont.continuation as cont

w_real = np.linspace(0., 5., num=2001, endpoint=True)
spec_real = np.exp(-(w_real)**2 / (2.*0.2**2))
spec_real += 0.3 * np.exp(-(w_real-1.5)**2 / (2.*0.8**2))
spec_real += 0.3 * np.exp(-(w_real+1.5)**2 / (2.*0.8**2))# must be symmetric around 0!
spec_real /= np.trapz(spec_real, w_real) # normalization

beta = 10.
iw = 2. * np.pi/beta * np.arange(10)

noise_amplitude = 1e-4 # create gaussian noise
rng = np.random.RandomState(1234)
noise = rng.normal(0., noise_amplitude, iw.shape[0])

kernel = (w_real**2)[None,:]/((iw**2)[:,None] + (w_real**2)[None,:])
kernel[0,0] = 1.
gf_bos = np.trapz(kernel*spec_real[None,:], w_real, axis=1) + noise
norm = gf_bos[0]
gf_bos /= norm

fig,ax = plt.subplots(ncols=2, nrows=1, figsize=(12,4))
ax[0].plot(w_real, spec_real, label='spectrum')
ax[0].plot(w_real, w_real*spec_real, label='response function')
ax[0].legend()
ax[0].set_xlabel('real frequency')
ax[0].set_title('Spectrum')
ax[1].plot(iw, gf_bos.real, marker='+')
ax[1].set_xlabel('Matsubara frequency')
ax[1].set_title('Matsubara Greensfunction')
plt.show()


w = np.linspace(0., 5., num=501, endpoint=True)
probl = cont.AnalyticContinuationProblem(im_axis=iw, re_axis=w,
                                        im_data=gf_bos, kernel_mode='freq_bosonic')

err = np.ones_like(iw) * noise_amplitude / norm
model = np.ones_like(w)
model /= np.trapz(model, w)
sol,_ = probl.solve(method='maxent_svd',
                    alpha_determination='chi2kink',
                    optimizer='newton',
                    stdev=err, model=model,
                    interactive=True)

np.save("A_opt_maxent_bosonic", sol.A_opt)
np.save("backtransform_maxent_bosonic", sol.backtransform)
np.save("chi2_maxent_bosonic", sol.chi2)

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15,4))
ax[0].plot(iw, gf_bos.real, linestyle='None', marker='+', label='data real')
ax[0].plot(iw, sol.backtransform.real, label='backtransform real')
ax[0].legend()
ax[0].set_xlabel('Matsubara frequency')
ax[0].set_title('Fit')
ax[1].plot(iw, gf_bos.real-sol.backtransform.real, label='real')
ax[1].legend()
ax[1].set_xlabel('Matsubara frequency')
ax[1].set_title('Misfit')
ax[2].plot(w, sol.A_opt, label='result spectrum')
ax[2].plot(w, w*sol.A_opt, label='result resp funct', linestyle='--')
ax[2].plot(w_real, spec_real, label='true spectrum')
ax[2].plot(w_real, w_real*spec_real, label='true resp funct', linestyle='--')
ax[2].legend()
ax[2].set_xlabel('real frequency')
ax[2].set_title('Spectrum')
plt.tight_layout()
plt.show()
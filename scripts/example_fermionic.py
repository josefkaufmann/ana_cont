import sys, os
import numpy as np
import matplotlib.pyplot as plt

file_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, package_dir)  # for applications, replace package_dir by the location of ana_cont
import ana_cont.continuation as cont

# real-frequency grid and example spectrum
w = np.linspace(-10., 10., num=501, endpoint=True)
spec = 0.4 * np.exp(-0.5 * (w - 1.8)**2) + 0.6 * np.exp(-0.5 * (w + 1.8)**2)
spec /= np.trapz(spec, w)

# Matsubara frequency grid and transformation of data
beta = 10.
niw = 20
iw = np.pi / beta * (2. * np.arange(niw) + 1.)
kernel = 1. / (1j * iw[:, None] - w[None, :])
giw = np.trapz(kernel * spec[None, :], w, axis=1)

# add noise to the data
rng = np.random.RandomState(4713)
noise_ampl = 0.0001
giw += noise_ampl * (rng.randn(niw) + 1j * rng.randn(niw))

# error bars and default model for analytic continuation
err = np.ones_like(iw) * noise_ampl
model = np.ones_like(w)
model /= np.trapz(model, w)

# specify the analytic continuation problem
probl = cont.AnalyticContinuationProblem(im_axis=iw, re_axis=w,
                                         im_data=giw, kernel_mode='freq_fermionic')

# solve the problem
sol, _ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton',
                     model=model, stdev=err, interactive=True, alpha_start=1e12, alpha_end=1e-2,
                     preblur=True, blur_width=0.5)

np.save("A_opt_maxent_fermionic", sol.A_opt)
np.save("backtransform_maxent_fermionic", sol.backtransform)
np.save("chi2_maxent_fermionic", sol.chi2)

# plot the results
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
ax[0].plot(w, spec, color='gray', label='real spectrum')
ax[0].plot(w, sol.A_opt, color='red', label='analytic continuation')
ax[0].set_xlabel('real frequency')
ax[0].legend()
ax[0].set_title('Spectrum')
ax[1].plot(iw, giw.real, marker='x', ls='None', label='data')
ax[1].plot(iw, giw.imag, marker='+', ls='None')
ax[1].plot(iw, sol.backtransform.real, color='red', ls='--', label='fit')
ax[1].plot(iw, sol.backtransform.imag, color='blue')
ax[1].set_xlabel('Matsubara frequency')
ax[1].legend()
ax[1].set_title('Data')
ax[2].plot(iw, (giw - sol.backtransform).real)
ax[2].plot(iw, (giw - sol.backtransform).imag)
ax[2].set_xlabel('Matsubara frequency')
ax[2].set_title('Data - Fit')
plt.tight_layout()
plt.show()
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

# specify the analytic continuation problem
mats_ind = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # Matsubara indices of data points to use in Pade
probl = cont.AnalyticContinuationProblem(im_axis=iw[mats_ind], re_axis=w,
                                         im_data=giw[mats_ind], kernel_mode='freq_fermionic')

# solve the problem
sol = probl.solve(method='pade')
check_axis = np.linspace(0., 1.25 * iw[mats_ind[-1]], num=500)
check = probl.solver.check(im_axis_fine=check_axis)

# plot the results
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))
ax[0].plot(w, spec, color='gray', label='real spectrum')
ax[0].plot(w, sol.A_opt, color='red', label='analytic continuation')
ax[0].set_xlabel('real frequency')
ax[0].legend()
ax[0].set_title('Spectrum')
ax[1].plot(iw[mats_ind], giw.real[mats_ind], marker='x', ls='None', label='data')
ax[1].plot(iw[mats_ind], giw.imag[mats_ind], marker='+', ls='None')
ax[1].plot(check_axis, check.real,
           ls='--', color='gray', label='Re[Pade interpolation]')
ax[1].plot(check_axis, check.imag,
           color='gray', label='Im[Pade interpolation]')
ax[1].set_xlabel('Matsubara frequency')
ax[1].legend()
ax[1].set_title('Data')
plt.tight_layout()
plt.show()
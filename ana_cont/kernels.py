import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integ


class Kernel(object):
    """This class handles the kernel of the analytic continuation."""

    def __init__(self, kind=None, re_axis=None, im_axis=None):
        if (kind is None
                or re_axis is None
                or im_axis is None):
            raise ValueError('Kernel not correctly initialized')
        self.kind = kind
        self.re_axis = re_axis
        self.im_axis = im_axis
        self.original_matrix = self.kernel_matrix()
        self.matrix = np.copy(self.original_matrix)
        self.nw = self.re_axis.shape[0]
        self.niw = self.im_axis.shape[0]

    def kernel_matrix(self):
        """Set the kernel"""
        if self.kind == 'freq_bosonic':
            kernel = (self.re_axis ** 2)[None, :] \
                     / ((self.re_axis ** 2)[None, :]
                        + (self.im_axis ** 2)[:, None])
            kernel[0, 0] = 1.  # analytically with de l'Hospital
        elif self.kind == 'time_bosonic':
            kernel = 0.5 * self.re_axis[None, :] * (
                    np.exp(-self.re_axis[None, :] * self.im_axis[:, None])
                    + np.exp(-self.re_axis[None, :] * (1. - self.im_axis[:, None]))) / (
                                  1. - np.exp(-self.re_axis[None, :]))
            kernel[:, 0] = 1.  # analytically with de l'Hospital
        elif self.kind == 'freq_bosonic_xyz':
            kernel = -self.im_axis[:, None] / ((self.re_axis**2)[None, :] + (self.im_axis**2)[:, None])
            if self.im_axis[0] == 0.:
                kernel[0] = 0.
        elif self.kind == 'freq_fermionic':
            kernel = 1. / (1j * self.im_axis[:, None] - self.re_axis[None, :])
        elif self.kind == 'time_fermionic':
            kernel = np.exp(-self.im_axis[:, None] * self.re_axis[None, :]) \
                          / (1. + np.exp(-self.re_axis[None, :]))
        elif self.kind == 'freq_fermionic_phsym':  # in this case, the data must be purely real (the imaginary part!)
            print('Warning: phsym kernels do not give good results in this implementation. ')
            kernel = -2. * self.im_axis[:, None]\
                     / ((self.im_axis ** 2)[:, None]
                        + (self.re_axis ** 2)[None, :])
        elif self.kind == 'time_fermionic_phsym':
            print('Warning: phsym kernels do not give good results in this implementation. ')
            kernel = (np.cosh(self.im_axis[:, None] * self.re_axis[None, :])
                      + np.cosh((1. - self.im_axis[:, None]) * self.re_axis[None, :])) \
                     / (1. + np.cosh(self.re_axis[None, :]))
        else:
            raise ValueError("Unknown kernel type")
        return kernel

    def preblur(self, blur_width):
        """Preblur for the kernel, if applicable."""
        self.blur_width = blur_width
        if self.blur_width > 0. and (self.kind == 'freq_fermionic' or self.kind == 'freq_bosonic'):
            self.matrix = self.convolve_kernel()
        else:
            self.matrix = np.copy(self.original_matrix)

    def convolve_kernel(self):
        """
        Convolve bosonic or fermionic kernel with a Gaussian.

        The Gaussian is g(x) = exp(x^2/(2b^2)) / (b sqrt(2 pi)).
        In the fermionic case, the convolution can be written as
        K_preblur(ivn, w) = \\int_{-5b}^{5b} dx g(x) / (ivn - x - w)

        In the bosonic case, the convolution can be written as
        K_preblur(iwn, w) = \\int_{-5b}^{5b} dx g(x) ((x+w)^2 / ((x+w)^2 + wn^2) + (x-w)^2 / ((x-w)^2 + wn^2)) / 2

        Integration over the Gaussian from -5b to 5b is certainly sufficient.
        Thus the Gaussian has to be computed only once and integration by scipy.integrate.simps
        gives very accurate results even for tiny values of b.

        Note: More direct calculation by, e.g., scipy.integrate.quad is possible but unstable
        and extremely slow.
        """

        self.w_integration = np.linspace(-5. * self.blur_width, 5. * self.blur_width, num=201, endpoint=True)
        norm = 1. / (self.blur_width * np.sqrt(2. * np.pi))
        self.gaussian_numeric = norm * np.exp(-0.5 * (self.w_integration / self.blur_width) ** 2)

        if self.kind == 'freq_fermionic':
            integrand = self.gaussian_numeric[None, None, :] \
                        / (1j * self.im_axis[:, None, None] - self.re_axis[None, :, None] - self.w_integration[None, None, :])
        elif self.kind == 'freq_bosonic':
            integrand_1 = self.gaussian_numeric[None, None, :] * (self.w_integration[None, None, :] + self.re_axis[None, :, None]) ** 2 \
                          / ((self.w_integration[None, None, :] + self.re_axis[None, :, None]) ** 2 + (self.im_axis[:, None, None]) ** 2)
            integrand_1[0] = self.gaussian_numeric[None, None, :]
            integrand_2 = self.gaussian_numeric[None, None, :] * (self.w_integration[None, None, :] - self.re_axis[None, :, None]) ** 2 \
                          / ((self.w_integration[None, None, :] - self.re_axis[None, :, None]) ** 2 + (self.im_axis[:, None, None]) ** 2)
            integrand_2[0] = self.gaussian_numeric[None, None, :]
            integrand = 0.5 * (integrand_1 + integrand_2)
        else:
            raise NotImplementedError('No preblur implemented for this kernel.')
        return integ.simps(integrand, x=self.w_integration, axis=-1)

    def blur(self, hidden_spectrum):
        """Convert hidden spectral function to spectral function."""
        if self.kind == 'freq_bosonic':
            h_interp = interp.InterpolatedUnivariateSpline(
                np.concatenate((-self.re_axis[:0:-1], self.re_axis)),
                np.concatenate((hidden_spectrum[:0:-1], hidden_spectrum)),
                ext='zeros')
        else:
            h_interp = interp.InterpolatedUnivariateSpline(
                self.re_axis,
                hidden_spectrum,
                ext='zeros')

        integrand = self.gaussian_numeric[None, :] * h_interp(self.re_axis[:, None] + self.w_integration[None, :])
        return integ.simps(integrand, x=self.w_integration, axis=-1)

    def real_matrix(self):
        """Return real and imaginary part one after another."""
        if self.kind == 'freq_fermionic':
            return np.concatenate((self.matrix.real, self.matrix.imag))
        else:
            return self.matrix

    def rotate_to_cov_eb(self, ucov):
        """Rotate first axis of kernel to eigenbasis of covariance matrix."""
        self.ucov = ucov
        self.matrix = np.dot(self.ucov.T.conj(), self.matrix)

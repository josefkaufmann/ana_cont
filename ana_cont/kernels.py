import numpy as np
import scipy.interpolate as interp

try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson


class Kernel(object):
    """This class handles the kernel of the analytic continuation."""

    def __init__(self, kind=None, re_axis=None, im_axis=None):
        """
        The kernel matrix is calculated from the real-frequency vector `re_axis`
        and the imaginary frequency or time vector `im_axis`.
        It is important to note that the Kernel does not know the inverse temperature
        beta. In case of the frequency kernels ('freq_fermionic' or 'freq_bosonic')
        the temperature is implicitly contained through the Matsubara frequencies.
        In case of the time kernels ('time_fermionic' or 'time_bosonic') the `im_axis`
        list has to be rescaled such that the temperature would be 1. Usually this is done
        by `imag_time = imag_time / beta`.

        For regular users of the library this is no problem, since the Kernel class is only
        instantiated inside the AnalyticContinuationProblem class, where the rescaling is done
        automatically. (Also the results are scaled back automatically there.)

        Parameters
        ----------
        kind : str, default=None
                  Which kind of kernel to use. Possible options:
                  'freq_bosonic', 'time_bosonic', 'freq_bosonic_xyz',
                  'freq_fermionic', 'time_fermionic', 'freq_fermionic_phsym',
                  'time_fermionic_phsym'
        re_axis : numpy.ndarray, default=None
                  real-frequency axis to generate the kernel matrix
        im_axis : numpy.ndarray, default=None
                  imaginary axis (time, Matsubara frequency)
                  to generate the kernel matrix
        """
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
        """Compute the kernel matrix.
        If you want to implement another kernel,
        you just have to add another 'elif' here. """

        if self.kind == 'freq_bosonic':
            with np.errstate(invalid="ignore"):
                kernel = (self.re_axis ** 2)[None, :] \
                         / ((self.re_axis ** 2)[None, :]
                            + (self.im_axis ** 2)[:, None])
            WhereIsiwn0 = np.where(self.im_axis==0.0)[0]
            WhereIsw0 = np.where(self.re_axis==0.0)[0]
            if len(WhereIsiwn0==1) and len(WhereIsw0==1):
                kernel[WhereIsiwn0, WhereIsw0] = 1.0 # analytically with de l'Hospital
        elif self.kind == 'time_bosonic':
            with np.errstate(invalid="ignore"):
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
            def time_kernel(τ, ω):
                """ 
                Kernel for fermions in imaginary time.
                Fixes problem for ω << 0
                
                Parameters:
                τ (float): imaginary time rescaled as τ/β in [0,1]
                ω (float): real frequency rescaled as ω*β 

                Returns:
                float: exp(-τ ω) /  (1 + exp(-ω))
               """    
                if np.exp(-ω) > 1.0E8:
                    return np.exp(-(τ-1)*ω)
                else:
                    return (np.exp(-τ * ω) /  (1. + np.exp(-ω)))
            time_kernel_vect = np.vectorize(time_kernel)
         
            kernel = time_kernel_vect(self.im_axis[:,None], self.re_axis[None,:])

            # OLD version has problems for very small ω  << 0!
            #kernel = np.exp(-self.im_axis[:, None] * self.re_axis[None, :]) \
            #              / (1. + np.exp(-self.re_axis[None, :]))
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

        The Gaussian is :math:`g(x) = \\frac{1}{b \\sqrt{2 \\pi}} exp(-x^2/(2b^2))`.
        In the fermionic case, the convolution can be written as
        :math:`K_{preblur}(i\\nu_n, \\omega) = \\int_{-5b}^{5b} dx\\; \\frac{g(x)}{i\\nu_n - x - \\omega}`

        In the bosonic case, the convolution can be written as
        :math:`K_{preblur}(i\\omega_n, \\nu) = \\frac{1}{2} \\int_{-5b}^{5b} dx\\; g(x) [\\frac{(x+\\nu)^2 }{ (x+\\nu)^2 + \\omega_n^2} + \\frac{(x-\\nu)^2 }{ (x-\\nu)^2 + \\omega_n^2}]`

        Integration over the Gaussian from :math:`-5b` to :math:`5b` is certainly sufficient.
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
            WhereIsiwn0 = np.where(self.im_axis==0.0)[0]
            with np.errstate(invalid="ignore"):
                integrand_1 = self.gaussian_numeric[None, None, :] * (self.w_integration[None, None, :] + self.re_axis[None, :, None]) ** 2 \
                              / ((self.w_integration[None, None, :] + self.re_axis[None, :, None]) ** 2 + (self.im_axis[:, None, None]) ** 2)
                integrand_2 = self.gaussian_numeric[None, None, :] * (self.w_integration[None, None, :] - self.re_axis[None, :, None]) ** 2 \
                              / ((self.w_integration[None, None, :] - self.re_axis[None, :, None]) ** 2 + (self.im_axis[:, None, None]) ** 2)
            if len(WhereIsiwn0) > 0:  # analytically with de l'Hospital
                integrand_1[WhereIsiwn0] = self.gaussian_numeric[None, None, :]
                integrand_2[WhereIsiwn0] = self.gaussian_numeric[None, None, :]
            integrand = 0.5 * (integrand_1 + integrand_2)
        else:
            raise NotImplementedError('No preblur implemented for this kernel.')
        return simpson(integrand, x=self.w_integration, axis=-1)

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
        return simpson(integrand, x=self.w_integration, axis=-1)

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

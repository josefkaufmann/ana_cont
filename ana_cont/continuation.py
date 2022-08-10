import sys
import numpy as np
if sys.version_info[0] > 2:
    from . import solvers
else:
    import solvers


class AnalyticContinuationProblem(object):
    """Specification of an analytic continuation problem.

    This class is designed to hold the information that specifies the analytic continuation problem:

    * Imaginary axis of the data
    * Real axis on which the result is anticipated
    * Imaginary-axis data (real-valued array)
    * Type of kernel (fermionic, bosonic, time, frequency)
    * Inverse temperature beta (only necessary for time kernels)

    Furthermore the class provides an interface to call the solver.
    """
    def __init__(self, im_axis=None, re_axis=None,
                 im_data=None, kernel_mode=None, beta=None):
        """
        Create instance of AnalyticContinuationProblem

        Parameters
        ----------
        im_axis : numpy.ndarray
                  One-dimensional numpy array of type float
                  Matsubara frequencies or imaginary time points of the input data.
        re_axis : numpy.ndarray
                  One-dimensional numpy array of type float
                  Real-frequency grid to use for the analytic continuation.
        im_data : numpy.ndarray
                  One-dimensional numpy array of type float or complex
                  Imaginary-axis data, e.g. Matsubara Green's function
        kernel_mode : {`'freq_fermionic'`, `'freq_bosonic'`, `'time_fermionic'`, `'time_bosonic'`}
                  * `'freq_fermionic'` fermionic Matsubara Greens function
                  * `'freq_bosonic'` bosonic Matsubara Greens function
                  * `'time_fermionic'` fermionic Green's function in imaginary time
                  * `'time_bosonic'` bosonic Green's function (susceptibility) in imaginary time
        beta :    float
                  Inverse temperature (only necessary for time kernels)
        """
        self.kernel_mode = kernel_mode
        if np.allclose(im_axis.imag, np.zeros_like(im_axis, dtype=float)):
            self.im_axis = im_axis
        else:
            raise ValueError("Parameter im_axis takes only the imaginary part of the imaginary axis (i.e. only real values)")
        self.re_axis = re_axis
        self.im_data = im_data
        if self.kernel_mode == 'freq_bosonic':
            pass # not necessary to do anything additionally here
        elif self.kernel_mode == 'time_bosonic':
            self.im_axis = im_axis/beta
            self.re_axis = re_axis*beta
            self.im_data = im_data*beta
            self.beta = beta
        elif self.kernel_mode == 'freq_fermionic':
            pass
            #self.im_data = np.concatenate((im_data.real, im_data.imag))
        elif self.kernel_mode == 'freq_fermionic_phsym':
            # check if data are purely imaginary
            if np.allclose(im_axis.real, 0.):
                self.im_data = im_data.imag
            elif np.allclose(im_axis.imag, 0.): # if only the imaginary part is passed
                self.im_data = im_data.real
            else:
                print('The data are neither purely real nor purely imaginary,')
                print('you cannot use a ph-symmetric kernel in this case.')
                sys.exit()
        elif self.kernel_mode == 'time_fermionic' \
                or self.kernel_mode == 'time_fermionic_phsym':
            self.im_axis = im_axis / beta
            self.re_axis = re_axis * beta
            self.im_data = im_data
            self.beta = beta
        else:
            raise ValueError("Unsupported kernel_mode.")


    def solve(self, method='', **kwargs):
        """ Interface function for solving the analytic continuation problem.

        :param method: Analytic continuation method, possible choices are 'maxent_svd', 'pade'
        :param kwargs: Further keyword arguments, specific to the solution method
        :return: OptimizationResult object

        This function first creates an instance of an AnalyticContinuationSolver object,
        then the respective solve function is called.
        """

        if method == 'maxent_svd':
            self.solver = solvers.MaxentSolverSVD(
                self.im_axis, self.re_axis, self.im_data,
                kernel_mode = self.kernel_mode, 
                **kwargs)
            sol = self.solver.solve(**kwargs)
            if self.kernel_mode == 'time_fermionic':
                sol[0].A_opt *= self.beta
            elif self.kernel_mode == 'time_bosonic':
                sol[0].A_opt *= self.beta
                sol[0].backtransform /= self.beta
            return sol
        elif method == 'maxent_mc':
            raise NotImplementedError
        elif method == 'pade':
            self.solver = solvers.PadeSolver(
                im_axis = self.im_axis, re_axis = self.re_axis,
                im_data = self.im_data)
            return self.solver.solve()
        else:
            raise ValueError("Unknown continuation method. Use one of 'maxent_svd', 'pade'.")

    def partial_solution(self, method='', **kwargs):
        """Maxent optimization at a specific value of alpha."""
        if method == 'maxent_svd':
            self.solver = solvers.MaxentSolverSVD(
                self.im_axis, self.re_axis, self.im_data,
                kernel_mode=self.kernel_mode,
                **kwargs)

            # truncate ustart to actual number of significant singular values
            kwargs['ustart'] = kwargs['ustart'][:self.solver.n_sv]
            sol = self.solver.maxent_optimization(**kwargs)
            if self.kernel_mode == 'time_fermionic':
                sol.A_opt *= self.beta
            elif self.kernel_mode == 'time_bosonic':
                sol.A_opt *= self.beta
                sol.backtransform /= self.beta
            return sol
        else:
            return ValueError("Unknown solver method.")


class GreensFunction(object):
    """
    This class defines a GreensFunction object. The main use of this
    is to calculate a full Green's function with real- and imaginary part
    from a spectrum.
    """

    def __init__(self, spectrum = None, wgrid = None, kind = ''):
        """Initialize with spectral function and real-frequency grid.

        Parameters
        ----------
        spectrum : np.ndarray
                   numpy array containing the spectral function
        wgrid :    np.ndarray
                   one-dimensional numpy array of type float
                   real-frequency values corresponding to the `spectrum`
        kind :     {`'fermionic'`, `'bosonic'`}
                   specify if the Green's function is fermionic or bosonic.
        """

        self.spectrum = spectrum
        self.wgrid = wgrid
        self.wmin = self.wgrid[0]
        self.wmax = self.wgrid[-1]
        self.dw = np.diff(
            np.concatenate(([self.wmin],
                            (self.wgrid[1:] + self.wgrid[:-1])/2.,
                            [self.wmax])))
        self.kind = kind # fermionic_phsym, bosonic,       fermionic
        #            or: symmetric,       antisymmetric, general

    def kkt(self):
        """Kramers Kronig transformation

        Obtain full complex Green's function from its imaginary part.
        """
        if self.kind == 'fermionic_phsym' or self.kind == 'symmetric':
            if self.wmin < 0.:
                print('warning: wmin<0 not permitted for fermionic_phsym greens functions.')
            with np.errstate(divide="ignore"):
                m = 2. * self.dw[:,None] * self.wgrid[:,None] * self.spectrum[:,None] \
                    / (self.wgrid[None,:]**2 - self.wgrid[:,None]**2)

        elif self.kind == 'bosonic' or self.kind == 'antisymmetric':
            if self.wmin < 0.:
                print('warning: wmin<0 not permitted for bosonic (antisymmetric) spectrum.')
            with np.errstate(divide="ignore"):
                m = 2. * self.dw[:,None] * self.wgrid[None,:] * self.spectrum[:,None]\
                    /(self.wgrid[None,:]**2 - self.wgrid[:,None]**2)
        
        elif self.kind == 'fermionic' or self.kind == 'general':
            with np.errstate(divide="ignore"):
                m = self.dw[:,None] * self.spectrum[:,None]\
                    /(self.wgrid[None,:]-self.wgrid[:,None])

        else:
            raise ValueError("Unknown kind of Greens function.")

        np.fill_diagonal(m, 0.)  # set manually where w==w'
        self.g_real = np.sum(m, axis=0)
        self.g_imag = -self.spectrum*np.pi
        return self.g_real + 1j*self.g_imag

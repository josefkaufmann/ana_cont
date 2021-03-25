import sys
import numpy as np
import scipy.interpolate as interp
if sys.version_info[0] > 2:
    from . import solvers
else:
    import solvers


class AnalyticContinuationProblem(object):
    def __init__(self, im_axis=None, re_axis=None,
                 im_data=None, kernel_mode=None, beta=None):
        self.kernel_mode = kernel_mode
        self.im_axis = im_axis
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


    def solve(self, method='', **kwargs):
        if method == 'maxent_svd':
            self.solver = solvers.MaxentSolverSVD(
                self.im_axis, self.re_axis, self.im_data,
                kernel_mode = self.kernel_mode, 
                **kwargs)
            sol = self.solver.solve(**kwargs)
            # TODO implement a postprocessing method, where the following should be done more carefully
            if self.kernel_mode == 'time_fermionic':
                sol[0].A_opt *= self.beta
            elif self.kernel_mode == 'time_bosonic':
                sol[0].A_opt *= self.beta
                sol[0].backtransform /= self.beta
            return sol
        if method == 'maxent_mc':
            raise NotImplementedError
        if method == 'pade':
            self.solver = solvers.PadeSolver(
                im_axis = self.im_axis, re_axis = self.re_axis,
                im_data = self.im_data)
            return self.solver.solve()

    def error_propagation(self,obs,args):
        return self.solver.error_propagation(obs,args)

    def partial_solution(self, method='', **kwargs):
        if method=='maxent_svd':
            self.solver=solvers.MaxentSolverSVD(
                self.im_axis, self.re_axis, self.im_data,
                kernel_mode=self.kernel_mode,
#                model=kwargs['model'], stdev=kwargs['stdev'], offdiag=kwargs['offdiag'])
                **kwargs)

            kwargs['ustart'] = kwargs['ustart'][:self.solver.n_sv]
            # sol = self.solver.maxent_optimization(kwargs['alpha'], ustart, **kwargs)
            sol = self.solver.maxent_optimization(**kwargs)
            if self.kernel_mode == 'time_fermionic':
                sol.A_opt *= self.beta
            elif self.kernel_mode == 'freq_fermionic':
                pass
#                bt = sol.backtransform
#                n = bt.shape[0] // 2
#                sol.backtransform = bt[:n] + 1j*bt[n:]
            elif self.kernel_mode == 'time_bosonic':
                sol.A_opt *= self.beta
                sol.backtransform /= self.beta
            return sol
        elif method=='maxent_plain':
            self.solver = solvers.MaxentSolverPlain(self.im_axis,
                                                    self.re_axis,
                                                    self.im_data,
                                                    kernel_mode=self.kernel_mode,
                                                    **kwargs)
            sol = self.solver.maxent_optimization(A_start=kwargs['model'], **kwargs)
            return sol
        else:
            return None

    def solve_preblur(self, verbose=False, interactive=False, **kwargs):
        """Solve the AnalyticContinuationProblem with preblur.

        First, determine the optimal alpha without preblur.
        Then, try to find a reasonable increment for the preblur search.
        The blur_width is increased, until chi2 is increased by a factor
        of 1.5 with respect to the no-preblur solution.
        One of the last elements, e.g. [-3] is taken as the final solution.
        """

        sol_noblur, sol_alphasearch = self.solve(method='maxent_svd',
                                                 optimizer='newton',
                                                 preblur=False,
                                                 alpha_determination='chi2kink',
                                                 verbose=verbose,
                                                 interactive=interactive,
                                                 **kwargs)

        # detect peaks to get good spacing for b
        spec_interp = interp.InterpolatedUnivariateSpline(self.re_axis, sol_noblur.A_opt, ext='zeros', k=4)
        extrema = spec_interp.derivative().roots()

        minimal_distance = np.amin(np.diff(extrema))
        b_spacing = 1. / (15. / minimal_distance + 200. / (self.re_axis[-1] - self.re_axis[0]))
        if verbose:
            print('Found extrema at {}'.format(extrema))
            print('Minimal distance between two extrema {}'.format(minimal_distance))
            print('Use spacing of {} for search of optimal blur'.format(b_spacing))
        b = 0.
        chi2_arr = [sol_noblur.chi2]
        b_arr = [0.]
        blur_solutions = []
        while True:
            b += b_spacing
            sol = self.partial_solution(method='maxent_svd',
                                        optimizer='newton',
                                        preblur=True,
                                        blur_width=b,
                                        verbose=verbose,
                                        alpha=sol_noblur.alpha,
                                        ustart=sol_noblur.u_opt,
                                        interactive=interactive,
                                        **kwargs)

            if sol.chi2 > 1.5 * sol_noblur.chi2 or len(b_arr) > 100:
                break
            blur_solutions.append(sol)
            b_arr.append(b)
            chi2_arr.append(sol.chi2)

        if interactive:
            import matplotlib.pyplot as plt
            plt.plot(b_arr, chi2_arr, marker='.', color='blue')
            plt.plot(b_arr[-3], chi2_arr[-3], marker='x', color='orange')
            plt.xlabel('b')
            plt.ylabel('chi2')
            plt.show()

        return blur_solutions[-3], [sol_alphasearch, blur_solutions]


# This class defines a GreensFunction object. The main use of this
# is to calculate a full Green's function with real- and imaginary part
# from a spectrum.
class GreensFunction(object):
    def __init__(self, spectrum = None, wgrid = None, kind = ''):
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
        if self.kind == 'fermionic_phsym' or self.kind == 'symmetric':
            if self.wmin < 0.:
                print('warning: wmin<0 not permitted for fermionic_phsym greens functions.')

            m = 2. * self.dw[:,None] * self.wgrid[:,None] * self.spectrum[:,None] \
                / (self.wgrid[None,:]**2 - self.wgrid[:,None]**2)

        elif self.kind == 'bosonic' or self.kind == 'antisymmetric':
            if self.wmin < 0.:
                print('warning: wmin<0 not permitted for bosonic (antisymmetric) spectrum.')
            m = 2. * self.dw[:,None] * self.wgrid[None,:] * self.spectrum[:,None]\
                /(self.wgrid[None,:]**2 - self.wgrid[:,None]**2)
        
        elif self.kind == 'fermionic' or self.kind == 'general':
            m = self.dw[:,None] * self.spectrum[:,None]\
                /(self.wgrid[None,:]-self.wgrid[:,None])

        np.fill_diagonal(m, 0.)
        self.g_real = np.sum(m, axis=0)
        self.g_imag = -self.spectrum*np.pi
        return self.g_real + 1j*self.g_imag

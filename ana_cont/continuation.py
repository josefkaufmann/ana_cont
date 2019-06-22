import sys
import numpy as np
if sys.version_info[0] > 2:
    from . import solvers
else:
    import solvers



def continue_maxent(im_data, stdev, kernel, beta, wmax,
                    wmin=0, nw=501, grid='linear', im_axis=None, model=None,
                    ):
    """
    Wrapper function that should simplify the analytic continuation.
    :param im_data: ndarray
        Data to continue. In case of scalar-valued Green's functions, im_data is a vector, for
        matrix-valued Green's functions, the last dimension corresponds to imaginary time/frequency.
    :param stdev: scalar/array
        Standard deviation of data. If stdev is a single number, a constant value is assumed at all
        frequency/time points. If it is a 1D array, it will be broadcasted to the same shape
        as im_data.
    :param kernel: string
        Kernel for analytic continuation.
        One of {'freq_bosonic', 'freq_fermionic', 'time_bosonic', 'time_fermionic'}.
    :param beta: scalar
        Inverse temperature of the problem. Necessary to create imaginary-axis grid and for time kernels.
    :param wmax: scalar
        Upper boundary of real-freqyency grid. Must be larger than wmin.
    :param wmin: scalar, optional
        Lower boundary of real-frequency grid. Must be 0 for bosonic kernels. Arbitrary for other kernels.
        Default WMIN=0.
    :param nw: integer, optional
        Number of grid points on real axis. Default NW=501
    :param grid: optional
        Can be a string or an array.
        If it is a string, it has to be one of {'linear', 'tangent'}. Default is 'linear'.
        'linear' generates a grid with regular spacing.
        'tangent' generates grid points according to
        \omega_k = \omega_{max} * \mathrm{tan(k \pi /(2.5 n_{\omega}))} / \mathrm{tan}(\pi/2.5)
    :param im_axis: vector, optional
        If given, im_axis holds the imaginary-time/Matsubara frequency grid on which im_data is given.
        If not present, the grid will be generated automatically.
        Automatically generated time-grids do NOT contain beta.
        Automatically generated fermionic Matsubara grids contain only positive frequencies.
        Automatically generated bosonic Matsubara grids contain positive frequencies, starting from 0.
    :param model, optional
        If not given, a constant default model will be generated.
        model='gauss': zero-centered gaussian default model with width (wmax-wmin)/6
        model='exp': exponential decay with length parameter (wmax-wmin)/3
        If a one-dimensional instance of np.ndarray is passed, it will be normalized and taken as default model.
        Note that the default model will be used only for diagonal elements of matrix-valued Green's functions.
    :return: maxent solution object
    """

    # generate imaginary grid
    if im_axis is None: # then we generate the imaginary grid
        if 'time' in kernel:
            im_axis = np.linspace(0.,beta, num=im_data.shape[-1], endpoint=False)
            print('automatically generated imaginary-time grid in [0, beta)')

        elif 'freq' in kernel:
            if 'bos' in kernel:
                im_axis = 2. * np.pi / beta * np.arange(im_data.shape[-1])
            elif 'ferm' in kernel:
                im_axis = np.pi / beta * (2. * np.arange(im_data.shape[-1]) + 1.)

    if isinstance(grid, np.ndarray):
        re_axis = grid
    # generate real grid
    elif 'bos' in kernel:
        if wmin != 0.:
            print('Warning: wmin is always 0 for bosonic kernels. '
            + 'Ignoring input wmin={}'.format(wmin))
        if 'lin' in grid:
            re_axis = np.linspace(0., wmax, num=nw, endpoint=True)
        elif 'tan' in grid:
            re_axis = wmax * np.tan(np.linspace(0.,np.pi/2.5, num=nw, endpoint=True))/np.tan(np.pi/2.5)
        else:
            raise ValueError('Unknown real frequency grid.')
    elif 'ferm' in kernel:
        if 'lin' in grid:
            re_axis = np.linspace(wmin, wmax, num=nw, endpoint=True)
        elif 'tan' in grid:
            if wmin > 0.:
                raise ValueError('A positive value of wmin does not make sense here. '
                + 'Use linear grid if you really want this.')
            re_neg = -wmin * np.tan(np.linspace(-np.pi/2.5, 0., num=nw//2, endpoint=False)) / np.tan(np.pi/2.5)
            re_pos = wmax * np.tan(np.linspace(0., np.pi/2.5, num=nw-nw//2, endpoint=True)) / np.tan(np.pi/2.5)
            re_axis = np.concatenate((re_neg, re_pos))

    # generate correct kernel name
    if 'bos' in kernel:
        if 'freq' in kernel:
            kernel_mode = 'freq_bosonic'
        elif 'time' in kernel:
            kernel_mode = 'time_bosonic'
        else:
            raise ValueError('Unknown kind of bosonic kernel')
    elif 'ferm' in kernel:
        if 'freq' in kernel:
            kernel_mode = 'freq_fermionic'
        elif 'time' in kernel:
            kernel_mode = 'time_fermionic'
        else:
            raise ValueError('Unknown kind of fermionic kernel')
    else:
        raise ValueError('Could not determine correct kernel mode from your input')

    if not isinstance(im_data, np.ndarray):
        raise TypeError('im_data has to be instance of np.ndarray')

    if im_data.ndim == 1:
        matrix_mode = False
    elif im_data.ndim ==3:
        matrix_mode = True
        n_orb = im_data.shape[0]
        if im_data.shape[0] != im_data.shape[1]:
            raise ValueError('Greens function must be quadratic in orbital space.')
        if not 'ferm' in kernel_mode:
            raise ValueError('Continuation of matrix-valued Greens functions '
                             + 'only possible with fermionic kernel.')
    else:
        raise ValueError('im_data must have either shape (NIW,) or (NORB, NORB, NIW)')


    if isinstance(stdev, np.ndarray):
        if matrix_mode and stdev.ndim == 1:
            err = stdev[None,None,:] * np.ones_like(im_data, dtype=np.float)
        elif matrix_mode and stdev.ndim == 3:
            err = stdev
        elif not matrix_mode and stdev.ndim == 1:
            err = stdev
        else:
            raise ValueError('Incorrectly formatted stdev.')
    else:
        err = np.ones_like(im_data) * stdev

    # generate a default model
    if isinstance(model, np.ndarray):
        print('user-specified default model')
        if matrix_mode:
            model_arr = np.zeros((n_orb, nw))
            if model.ndim == 1:
                for i in range(n_orb):
                    model_arr[i] = model
            elif model.ndim ==2:
                for i in range(n_orb):
                    model_arr[i] = model[i]
        else:
            model_arr = model
    elif model is None or 'const' in model:
        print('generate constant default model')
        model_arr = np.ones_like(re_axis)
    elif 'gauss' in model:
        print('gaussian default model')
        width = (wmax - wmin) / 6.
        model_arr = np.exp(-re_axis**2 / (2. * width**2))
    elif 'exp' in model:
        print('default model with exponential decay')
        width = (wmax - wmin) / 3.
        model_arr = np.exp(-re_axis/width)
    else:
        raise ValueError('Default model specification not recognized.')

    # normalize the model
    if not matrix_mode:
        model_arr /= np.trapz(model_arr, re_axis)
    else:
        for i in range(n_orb):
            model_arr[i] /= np.trapz(model_arr[i], re_axis)

    if not matrix_mode:
        if kernel_mode == 'freq_bosonic':
            norm = im_data[0].real
        elif kernel_mode == 'time_bosonic':
            norm = np.trapz(im_data, im_axis).real
        else:
            norm = 1.

        probl = AnalyticContinuationProblem(im_axis=im_axis, re_axis=re_axis,
                                            im_data=im_data/norm, kernel_mode=kernel_mode,
                                            beta=beta)
        sol,_ = probl.solve(method='maxent_svd', alpha_determination='classic',
                            stdev = stdev/norm, model=model_arr,
                            preblur=False)

        sol.A_opt *= norm
        sol.backtransform *= norm
        sol.norm *= norm

        return sol

    else:
        sol_diag=[]
        for i in range(n_orb):
            print('Continuing component ({},{})'.format(i,i))
            probl = AnalyticContinuationProblem(im_axis=im_axis, re_axis=re_axis,
                                                im_data=im_data[i,i], kernel_mode=kernel_mode)
            sol_diag.append(
                probl.solve(method='maxent_svd',
                            model=model_arr[i],
                            stdev=err[i,i],
                            alpha_determination='classic',
                            offdiag=False, preblur=False)[0])

        sol_offd = []
        for i in range(n_orb):
            sol_offd.append([])
            for j in range(n_orb):
                print('Continuing component ({},{})'.format(i,j))
                dat = im_data[i,j]
                if np.any(np.abs(dat)>0.00001) and np.all(np.isfinite(dat)) and i != j:
                    model_offd = np.sqrt(sol_diag[i].A_opt * sol_diag[j].A_opt)
                    probl = AnalyticContinuationProblem(im_axis=im_axis, re_axis=re_axis,
                                                             im_data=dat, kernel_mode=kernel_mode)
                    sol_offd[i].append(
                        probl.solve(method='maxent_svd',
                                    model=model_offd,
                                    stdev=err[i,j],
                                    alpha_determination='classic',
                                    offdiag=True, preblur=False)[0])
                else:
                    sol_offd[i].append(None)

        for i in range(n_orb):
            sol_offd[i][i] = sol_diag[i]

        return sol_offd


class AnalyticContinuationProblem(object):
    def __init__(self, im_axis=None, re_axis=None,
                 im_data=None, kernel_mode=None, beta=1.):
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

    def partial_solution(self,method='',**kwargs):
        if method=='maxent_svd':
            self.solver=solvers.MaxentSolverSVD(
                self.im_axis, self.re_axis, self.im_data,
                kernel_mode=self.kernel_mode,
#                model=kwargs['model'], stdev=kwargs['stdev'], offdiag=kwargs['offdiag'])
                **kwargs)
            ustart=kwargs['ustart'][:self.solver.n_sv]
            sol = self.solver.maxent_optimization(kwargs['alpha'],ustart)
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
        else:
            return None


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

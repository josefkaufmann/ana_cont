import sys
import numpy as np
import scipy.optimize as opt
import collections
from abc import ABC, abstractmethod
from . import kernels

if sys.version_info[0] > 2:
    try:
        from . import pade
    except:
        pass
else:
    try:
        import pade
    except:
        pass


class AnalyticContinuationSolver(ABC):
    """Abstract base class for solver classes.

    Each Solver class has to inherit from this class.
    The purpose of this is to ensure that each child class
    has a method 'solve' implemented.
    """
    @abstractmethod
    def solve(self):
        pass


class PadeSolver(AnalyticContinuationSolver):
    """Pade solver"""
    def __init__(self, im_axis, re_axis, im_data):
        """
        Parameters
        ----------
        im_axis : numpy.ndarray
                  Matsubara frequencies which are used for the continuation
        re_axis : numpy.ndarray
                  Real-frequency points at which the Pade interpolant is evaluated
        im_data : Green's function values at the given points `im_axis`
        """
        self.im_axis = im_axis
        self.re_axis = re_axis
        self.im_data = im_data

        # Compute the Pade coefficients
        self.a = pade.compute_coefficients(1j * self.im_axis, self.im_data)

    def check(self, im_axis_fine=None):
        """Sanity check for Pade approximant

        Evaluate the Pade approximant on the imaginary axis,
        however not only at Matsubara frequencies, but on a
        dense grid. If the approximant is good, then this
        should yield a smooth interpolating curve on the Matsubara
        axis. On the other hand, little 'spikes' are a hint
        for pole-zero pairs close to the imaginary axis. This
        is usually a result of noise and a different choice of
        Matsubara frequencies may help.

        Parameters
        ----------
        im_axis_fine : numpy.ndarray, default=None
                  Imaginary-axis points where the approximant is
                  evaluated for checking. If not specified,
                  an array of length 500 is generated, which goes
                  up to twice the maximum frequency that was used
                  for constructing the approximant.

        Returns
        -------
        numpy.ndarray
                  Values of the Pade approximant at the points
                  of `im_axis_fine`.
        """

        if im_axis_fine is None:
            self.ivcheck = np.linspace(0, 2 * np.max(self.im_axis), num=500)
        else:
            self.ivcheck = im_axis_fine
        check = pade.C(self.ivcheck * 1j, 1j * self.im_axis,
                            self.im_data, self.a)
        return check

    def solve(self, show_plot=False):
        """Compute the Pade approximation on the real axis.

        The main numerically heavy computation is done in the Cython module
        `pade.pyx`. Here we just call the functions.
        In the Pade method, the numerator and denominator approximants
        are generated separately, and then the division is done.
        As an additional feature we add the callable `numerator_function`
        and `denominator_function` to the OptimizationResult object.
        """

        def numerator_function(z):
            return pade.A(z, self.im_axis.shape[0],
                           1j * self.im_axis, self.im_data, self.a)

        def denominator_function(z):
            return pade.B(z, self.im_axis.shape[0],
                           1j * self.im_axis, self.im_data, self.a)

        numerator = pade.A(self.re_axis, self.im_axis.shape[0],
                           1j * self.im_axis, self.im_data, self.a)
        denominator = pade.B(self.re_axis, self.im_axis.shape[0],
                           1j * self.im_axis, self.im_data, self.a)

        result = numerator / denominator

        result_dict = {}
        result_dict.update({"numerator": numerator,
                            "denominator": denominator,
                            "numerator_function": numerator_function,
                            "denominator_function": denominator_function,
                            "check": self.check(),
                            "ivcheck": self.ivcheck,
                            "A_opt": -result.imag / np.pi,
                            "g_ret": result})

        return OptimizationResult(**result_dict)


class MaxentSolverSVD(AnalyticContinuationSolver):
    """Maxent with singular-value decomposition.

    The singular-value decomposition of the kernel leads to a basis
    in which the dimensionality of the analytic continuation problem
    is much reduced. This makes computations faster and enforces some
    constraints.

    This class is never instantiated directly by the user, but instead by
    the solve method of continuation.AnalyticContinuationProblem.
    """

    def log(self, msg):
        """Print log messages if `verbose=True` was passed to the solver.

        Parameters
        ----------
        msg : str
                  Message to print
        """

        if self.verbose: print(msg)

    def __init__(self, im_axis, re_axis, im_data,
                 kernel_mode='', model=None,
                 stdev=None, cov=None,
                 offdiag=False,
                 preblur=False, blur_width=0.,
                 optimizer='newton',
                 verbose=True, **kwargs):
        """Create a Maxent solver object.

        Here we pass the data to the solver and do some precomputations.
        This makes the process of solving much faster.

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

                  Additionally there are less established special-purpose options
                  [`'freq_bosonic_xyz'`, `'freq_fermionic_phsym'`, `'time_fermionic_phsym'`].
        model : numpy.ndarray
                  One-dimensional numpy array of type float.
                  (Values must be greater or equal to zero.)
                  Default model of the Maxent calculation, i.e. the entropy of the
                  spectral function is computed with respect to this model.
                  Thus the shape has to match `re_axis`.
        stdev : numpy.ndarray
                  One-dimensional numpy array of positive float values.
                  `stdev` are the standard deviation values of the measured data points,
                  thus they have to be larger than zero. The shape has to match `im_data`
                  The keywords `stdev` and `cov` are mutually exclusive, i.e. only one of them
                  can be used.
                  For complex data (e.g. Matsubara Green's function) the same standard deviation
                  is used for both the real and imaginary part, such that the array passed to this
                  keyword is taken as the standard deviation of the real part (not the absolute).
        cov : numpy.ndarray
                  Two-dimensional numpy array.
                  `cov` is the covariance matrix of the input data. If it is diagonal, the diagonal
                  elements are the variance of the data, i.e. the square of the standard deviation.
                  The keywords `stdev` and `cov` are mutually exclusive, i.e. only one of them
                  can be used.
                  For complex data, the covariance matrix can be complex (see Kappl et al., PRB 102 (8), 085124)
        offdiag : bool, default: False
                  Whether the input data are offdiagonal elements of a matrix. For diagonal elements,
                  the spectral function is positive, for offdiagonal ones it has positive and negative
                  values, and the integral is 0.
        preblur : bool, default: False
                  Whether to use preblur in the calculation
        blur_width : float, default: 0.
                  If preblur=True, specify a width of the preblur that is larger than 0.
        optimizer : {'scipy_lm', 'newton'}, default: 'newton'
                  Which optimizer to use at each value of alpha to solve the minimization / root finding problem.
                  Usually the custom implementation of the newton algorithm is much faster.
        verbose : bool, default: True
                  If set to False, no output is printed. This can be useful when doing a very large number
                  of computations, but generally I recommend leaving it on True, to see what is happening.

        """

        self.verbose = verbose

        self.kernel_mode = kernel_mode
        if np.allclose(im_axis.imag, np.zeros_like(im_axis, dtype=float)):
            self.im_axis = im_axis
        else:
            raise ValueError("Parameter im_axis takes only the imaginary part of the imaginary axis (i.e. only real values)")
        self.re_axis = re_axis
        self.im_data = im_data
        self.offdiag = offdiag
        self.optimizer = optimizer

        self.nw = self.re_axis.shape[0]
        self.wmin = self.re_axis[0]
        self.wmax = self.re_axis[-1]
        self.dw = np.diff(
            np.concatenate(([self.wmin],
                            (self.re_axis[1:] + self.re_axis[:-1]) / 2.,
                            [self.wmax])))
        if not self.offdiag:
            self.model = model  # the model should be normalized by the user himself
        else:
            self.model_plus = model  # the model should be normalized by the user himself
            self.model_minus = model  # the model should be normalized by the user himself

        if cov is None and stdev is not None:
            self.var = stdev ** 2
            self.cov = np.diag(self.var)
            self.ucov = np.eye(self.im_axis.shape[0])
        elif stdev is None and cov is not None:
            self.cov = cov
            self.var, self.ucov = np.linalg.eigh(self.cov)  # go to eigenbasis of covariance matrix
            self.var = np.abs(self.var)  # numerically, var can be zero below machine precision

        self.im_data = np.dot(self.ucov.T.conj(), self.im_data)
        self.E = 1. / self.var
        self.niw = self.im_axis.shape[0]

        # set the kernel
        self.kernel = kernels.Kernel(kind=kernel_mode,
                                     im_axis=self.im_axis,
                                     re_axis=self.re_axis)

        # PREBLUR
        self.preblur = preblur
        if self.preblur:
            self.kernel.preblur(blur_width=blur_width)

        # rotate kernel to eigenbasis of covariance matrix
        self.kernel.rotate_to_cov_eb(ucov=self.ucov)

        # special treatment for complex data of fermionic frequency kernel
        if kernel_mode == 'freq_fermionic':
            self.niw *= 2
            self.im_data = np.concatenate((self.im_data.real, self.im_data.imag))
            self.var = np.concatenate((self.var, self.var))
            self.E = np.concatenate((self.E, self.E))

        # singular value decomposition of the kernel
        U, S, Vt = np.linalg.svd(self.kernel.real_matrix(), full_matrices=False)

        self.n_sv = np.arange(min(self.nw, self.niw))[S > 1e-10][-1]  # number of singular values larger than 1e-10

        self.U_svd = np.array(U[:, :self.n_sv], dtype=np.float64, order='C')
        self.V_svd = np.array(Vt[:self.n_sv, :].T, dtype=np.float64, order='C')  # numpy.svd returns V.T
        self.Xi_svd = S[:self.n_sv]

        self.log('{} data points on real axis'.format(self.nw))
        self.log('{} data points on imaginary axis'.format(self.niw))
        self.log('{} significant singular values'.format(self.n_sv))

        # =============================================================================================
        # First, precompute as much as possible
        # The precomputation of W2 is done in C, this saves a lot of time!
        # The other precomputations need less loop, can stay in python for the moment.
        # =============================================================================================
        self.log('Precomputation of coefficient matrices...')

        if not self.offdiag:  # precompute matrices W_ml (W2), W_mil (W3)
            self.W2 = np.einsum('k,km,m,kn,n,ln,l,l->ml', self.E, self.U_svd, self.Xi_svd, self.U_svd, self.Xi_svd,
                                self.V_svd, self.dw, self.model)
            self.W3 = self.W2[:, None, :] * (self.V_svd[None, :, :]).transpose((0, 2, 1))

        else:  # precompute matrices M_ml (M2), M_mil (M3)
            self.M2 = np.einsum('k,km,m,kn,n,ln,l->ml', self.E, self.U_svd, self.Xi_svd, self.U_svd, self.Xi_svd,
                                self.V_svd, self.dw)
            self.M3 = self.M2[:, None, :] * (self.V_svd[None, :, :]).transpose((0, 2, 1))

        # precompute the evidence vector Evi_m
        self.Evi = np.einsum('m,km,k,k->m', self.Xi_svd, self.U_svd, self.E, self.im_data)

        # precompute curvature of likelihood function
        self.d2chi2 = np.einsum('i,j,ki,kj,k->ij', self.dw, self.dw,
                                self.kernel.real_matrix(), self.kernel.real_matrix(),
                                self.E)

        # some arrays that are used later...
        self.chi2arr = []
        self.specarr = []
        self.backarr = []
        self.entrarr = []
        self.alpharr = []
        self.uarr = []
        self.bayesConv = []

    # =============================================================================================
    # Here, we define the main functions needed for the root finding problem
    # =============================================================================================

    def compute_f_J_diag(self, u, alpha):
        """This function evaluates the function whose root we want to find.

        The function :math:`f_m(u)` is defined as the singular value decomposition
        of the derivative :math:`dQ[A]/dA_m`. Since we want to minimize :math:`Q[A]`,
        we have to find the root of the vector-valued function :math:`f`, i.e.
        :math:`f_m(u) = SVD(dQ/dA)_m = 0`.
        For more efficient root finding, we also need the Jacobian :math:`J`.
        It is directly computed in singular space, :math:`J_{mi}=df_m/du_i`.

        Parameters
        ----------
        u : numpy.ndarray
                  singular-space vector that parametrizes the spectral function
        alpha : float
                  (positive) weight factor of the entropy

        Returns
        -------
        f : numpy.ndarray
                  value of the function whose zero we want to find
        J : numpy.ndarray
                  Jacobian at the current position
        """

        v = np.dot(self.V_svd, u)
        w = np.exp(v)
        term_1 = np.dot(self.W2, w)
        term_2 = np.dot(self.W3, w)
        f = alpha * u + term_1 - self.Evi
        J = alpha * np.eye(self.n_sv) + term_2
        return f, J

    def compute_f_J_offdiag(self, u, alpha):
        """The analogue to compute_f_J_diag for offdiagonal elements.

        Parameters
        ----------
        u : numpy.ndarray
                  singular-space vector that parametrizes the spectral function
        alpha : float
                  (positive) weight factor of the entropy

        Returns
        -------
        f : numpy.ndarray
                  value of the function whose zero we want to find
        J : numpy.ndarray
                  Jacobian at the current position
        """
        v = np.dot(self.V_svd, u)
        w = np.exp(v)
        a_plus = self.model_plus * w
        a_minus = self.model_minus / w
        a1 = a_plus - a_minus
        a2 = a_plus + a_minus
        f = alpha * u + np.dot(self.M2, a1) - self.Evi
        J = alpha * np.eye(self.n_sv) + np.dot(self.M3, a2)
        return f, J

    # =============================================================================================
    # Some auxiliary functions
    # =============================================================================================

    def singular_to_realspace_diag(self, u):
        """Go from singular to real space.

        Transform the singular space vector u into real-frequency space (spectral function)
        by :math:`A(\\omega) = D(\\omega) e^{V u}`, where :math:`D` is the default model
        and :math:`V` is the matrix from the SVD.

        Parameters
        ----------
        u : numpy.ndarray
                  singular-space vector that parametrizes the spectral function

        Returns
        -------
        numpy.ndarray
                  Spectral function :math:`A` obtained from `u`
        """
        return self.model * np.exp(np.dot(self.V_svd, u))

    def singular_to_realspace_offdiag(self, u):
        """Go from singular to real space.

        Transform the singular space vector u into real-frequency
        space in the case of an offdiagonal element.

        Parameters
        ----------
        u : numpy.ndarray
                  singular-space vector that parametrizes the spectral function

        Returns
        -------
        numpy.ndarray
                  Spectral function :math:`A` obtained from `u`
        """
        v = np.dot(self.V_svd, u)
        w = np.exp(v)
        return self.model_plus * w - self.model_minus / w

    def backtransform(self, A):
        """ Backtransformation from real to imaginary axis.

        :math:`G(i\\omega_n) = \int d\\nu \\; K(i\\omega_n, \\nu)  A(\\nu)`

        Also at this place we return from the 'diagonal-covariance space'
        Note: this function is not a bottleneck.

        Parameters
        ----------
        A : numpy.ndarray
                  Spectral function

        Returns
        -------
        np.ndarray
                  Back-transformed Green's function on imaginary axis
        """
        return np.trapz(np.dot(self.ucov, self.kernel.matrix) * A[None, :],
                        self.re_axis, axis=-1)

    def chi2(self, A):
        """compute chi-squared-deviation

        Compute the log-likelihood function or chi-squared-deviation of
        the spectral function:
        :math:`\\sum_n \\frac{|G(i\\omega_n) - \int K(i\\omega_n, \\nu) A(\\nu)|^2}{\\sigma_n^2}`

        Parameters
        ----------
        A : numpy.ndarray
                  Spectral function

        Returns
        -------
        float
                  chi-squared deviation
        """
        return np.sum(
            self.E * (self.im_data - np.trapz(self.kernel.real_matrix() * A[None, :], self.re_axis, axis=-1)) ** 2)

    def entropy_pos(self, A, u):
        """Compute entropy for positive definite spectral function.

        Parameters
        ----------
        A : numpy.ndarray
                  Spectral function
        u : numpy.ndarray
                  Singular-space vector representing the same spectral function

        Returns
        -------
        float
                  entropy"""
        return np.trapz(A - self.model - A * np.dot(self.V_svd, u), self.re_axis)

    def entropy_posneg(self, A, u):
        """Compute "positive-negative entropy" for spectral function with norm 0.

        Parameters
        ----------
        A : numpy.ndarray
                  Spectral function
        u : numpy.ndarray
                  Singular-space vector representing the same spectral function

        Returns
        -------
        float
                  entropy
        """
        root = np.sqrt(A ** 2 + 4. * self.model_plus * self.model_minus)
        return np.trapz(root - self.model_plus - self.model_minus
                        - A * np.log((root + A) / (2. * self.model_plus)),
                        self.re_axis)

    def bayes_conv(self, A, entr, alpha):
        """Bayesian convergence criterion for classic maxent (maximum of probablility distribution)

        Parameters
        ----------
        A : numpy.ndarray
                  spectral function
        entr : float
                  entropy
        alpha : float
                  weight factor of the entropy

        Returns
        -------
        ng : float
                  "number of good data points"
        tr : float
                  trace of the gamma matrix
        conv : float
                   convergence criterion (1 -> converged)
        """

        LambdaMatrix = np.sqrt(A / self.dw)[:, None] * self.d2chi2 * np.sqrt(A / self.dw)[None, :]
        try:
            lam = np.linalg.eigvalsh(LambdaMatrix)
        except np.linalg.LinAlgError:
            self.log('LinAlgError in Bayes matrix inversion. Irrelevant for chi2kink')
            lam = np.diag(LambdaMatrix)
        ng = -2. * alpha * entr
        tr = np.sum(lam / (alpha + lam))
        conv = tr / ng
        return ng, tr, conv

    def bayes_conv_offdiag(self, A, entr, alpha):
        """Bayesian convergence criterion for classic maxent, offdiagonal version

        Parameters
        ----------
        A : numpy.ndarray
                  spectral function
        entr : float
                  entropy
        alpha : float
                  weight factor of the entropy

        Returns
        -------
        ng : float
                  "number of good data points"
        tr : float
                  trace of the gamma matrix
        conv : float
                   convergence criterion (1 -> converged)
        """
        A_sq = np.power((A ** 2 + 4. * self.model_plus * self.model_minus) / self.dw ** 2, 0.25)
        LambdaMatrix = A_sq[:, None] * self.d2chi2 * A_sq[None, :]
        lam = np.linalg.eigvalsh(LambdaMatrix)
        ng = -2. * alpha * entr
        tr = np.sum(lam / (alpha + lam))
        conv = tr / ng
        return ng, tr, conv

    def posterior_probability(self, A, alpha, entr, chisq):
        """Bayesian a-posteriori probability for alpha after optimization of A

        Parameters
        ----------
        A : numpy.ndarray
                  spectral function
        entr : float
                  entropy
        alpha : float
                  weight factor of the entropy
        chisq : float
                  chi-squared deviation

        Returns
        -------
        float
                 Probability
        """
        lambda_matrix = np.sqrt(A / self.dw)[:, None] * self.d2chi2 * np.sqrt(A / self.dw)[None, :]
        try:
            lam = np.linalg.eigvalsh(lambda_matrix)
        except np.linalg.LinAlgError:
            self.log('LinAlgError in Bayes matrix inversion. Irrelevant for chi2kink')
            lam = np.diag(lambda_matrix)
        try:
            eig_sum = np.sum(np.log(alpha / (alpha + lam)))
        except RuntimeWarning:
            print(lam)
        log_prob = alpha * entr - 0.5 * chisq + np.log(alpha) + 0.5 * eig_sum
        return np.exp(log_prob)

    def maxent_optimization(self, alpha, ustart, iterfac=10000000, use_bayes=False, **kwargs):
        """optimization of maxent functional for a given value of alpha

        Since a priori the best value of :math:`\\alpha` is unknown,
        this function has to be called several times in order to find a good value.

        Parameters
        ----------
        alpha : float
                  weight factor of the entropy
        ustart : numpy.ndarray
                  singular-space vector used as a starting value for the optimization.
                  For the first optimization, done at large alpha, we use zeros,
                  which corresponds to the default model. Then we use the result of the
                  previous optimization as a starting value.
        iterfac : int, default: 10000000
                  Control parameter for maximum number of iterations in
                  scipy_lm, which is <number of singular values> * iterfac.
                  It has no effect when using the newton optimizer.
        use_bayes : bool, default=False
                  Whether to use the Bayesian inference parameters for alpha.

        Returns
        -------
        OptimizationResult
                  Object that holds the results of the optimization,
                  e.g. spectral function, chi-squared deviation.

        """

        if not self.offdiag:
            self.compute_f_J = self.compute_f_J_diag
            self.singular_to_realspace = self.singular_to_realspace_diag
            self.entropy = self.entropy_pos
        else:
            self.compute_f_J = self.compute_f_J_offdiag
            self.singular_to_realspace = self.singular_to_realspace_offdiag
            self.entropy = self.entropy_posneg

        if self.optimizer == 'newton':
            newton_solver = NewtonOptimizer(self.n_sv, initial_guess=ustart, max_hist=1)
            sol = newton_solver(self.compute_f_J, alpha)
        elif self.optimizer == 'scipy_lm':
            sol = opt.root(self.compute_f_J,  # function returning function value f and jacobian J (we search root of f)
                          ustart,  # sensible starting point
                          method='lm',  # levenberg-marquart method
                          jac=True,  # already self.compute_f_J returns the jacobian (slightly more efficient in this case)
                          options={'maxiter': iterfac * self.n_sv,  # max number of lm steps
                                   'factor': 100.,  # scale for initial stepwidth of lm (?)
                                   'diag': np.exp(np.arange(self.n_sv))},
                          # scale for values to find (assume that they decay exponentially)
                          args=(alpha))  # additional argument for self.compute_f_J
        else:
            raise ValueError("Unknown optimizer. Use either 'newton' (recommended) or 'scipy_lm'.")

        u_opt = sol.x
        A_opt = self.singular_to_realspace(sol.x)
        entr = self.entropy(A_opt, u_opt)
        chisq = self.chi2(A_opt)  # has to be applied before blurring
        norm = np.trapz(A_opt, self.re_axis)  # is not changed by blurring

        # result = OptimizationResult()
        result_dict = {}
        result_dict.update({"u_opt": u_opt})
        if self.preblur:
            result_dict.update({"A_opt": self.kernel.blur(A_opt),
                                "blur_width": self.kernel.blur_width})
            # result.A_opt = self.kernel.blur(A_opt)
            # result.blur_width = self.kernel.blur_width
        else:
            result_dict.update({"A_opt": A_opt, "blur_width": 0.})
            # result.A_opt = A_opt
            # result.blur_width = 0.
        result_dict.update({"alpha": alpha,
                            "entropy": entr,
                            "chi2": chisq,
                            "backtransform": self.backtransform(A_opt),
                            "norm": norm,
                            "Q": alpha * entr - 0.5 * chisq})
        # result.alpha = alpha
        # result.entropy = entr
        # result.chi2 = chisq
        # result.backtransform = self.backtransform(A_opt)
        # result.norm = norm
        # result.Q = alpha * entr - 0.5 * chisq

        if use_bayes:
            if not self.offdiag:
                ng, tr, conv = self.bayes_conv(A_opt, entr, alpha)
            else:
                ng, tr, conv = self.bayes_conv_offdiag(A_opt, entr, alpha)

            # result.n_good = ng
            # result.trace = tr
            # result.convergence = conv
            result_dict.update({"n_good": ng, "trace": tr, "convergence": conv})
            if not self.offdiag:
                prob = self.posterior_probability(A_opt, alpha, entr, chisq)
                # result.probability = prob
                result_dict.update({"probability": prob})

        self.log('log10(alpha) = {:3.2f},\tchi2 = {:4.3e},   S = {:4.3e},   nfev = {},   norm = {:4.3f}'.format(
            np.log10(alpha), chisq, entr, sol.nfev, norm))

        return OptimizationResult(**result_dict)

    # =============================================================================================
    # Several variants of maxent are implemented in the following
    # =============================================================================================

    def solve_historic(self):
        """ Historic Maxent: choose alpha in a way that chi^2 \approx N

        Returns
        -------
        OptimizationResult
               Result of the optimization at "best" alpha value
        list
               List of OptimizationResult objects for all used values of alpha
        """

        if not self.offdiag:
            self.compute_f_J = self.compute_f_J_diag
            self.singular_to_realspace = self.singular_to_realspace_diag
            self.entropy = self.entropy_pos
        else:
            self.compute_f_J = self.compute_f_J_offdiag
            self.singular_to_realspace = self.singular_to_realspace_offdiag
            self.entropy = self.entropy_posneg
        self.log('Solving...')
        optarr = []
        alpha = 10 ** 6
        self.ustart = np.zeros((self.n_sv))
        converged = False
        conv = 0.
        while conv < 1:
            o = self.maxent_optimization(alpha, self.ustart)
            optarr.append(o)
            alpha /= 10.
            conv = self.niw / o.chi2

        if len(optarr) <= 1:
            raise RuntimeError('Failed to get a prediction for optimal alpha. '
                               + 'Decrease the error or increase alpha_start.')

        def root_fun(alpha, u0):  # this function is just for the newton root-finding
            res = self.maxent_optimization(alpha, u0, iterfac=100000)
            optarr.append(res)
            u0[:] = res.u_opt
            return self.niw / res.chi2 - 1.

        ustart = optarr[-2].u_opt
        alpha_opt = opt.newton(root_fun, optarr[-1].alpha, tol=1e-6, args=(ustart,))
        self.log('final optimal alpha: {}, log10(alpha_opt) = '.format(alpha_opt, np.log10(alpha_opt)))

        sol = self.maxent_optimization(alpha_opt, ustart, iterfac=250000)
        self.alpha_opt = alpha_opt
        self.A_opt = sol.A_opt
        return sol, optarr


    def solve_classic(self):
        """Classic maxent alpha determination.

        Classic maxent uses Bayes statistics to approximately determine
        the most probable value of alpha
        We start at a large value of alpha, where the optimization yields basically the default model,
        therefore u_opt is only a few steps away from ustart=0 (=default model)
        Then we gradually decrease alpha, step by step moving away from the default model towards data fitting.
        Using u_opt as ustart for the next (smaller) alpha brings a great speedup into this procedure.

        Returns
        -------
        OptimizationResult
               Result of the optimization at "best" alpha value
        list
               List of OptimizationResult objects for all used values of alpha
        """
        if not self.offdiag:
            self.compute_f_J = self.compute_f_J_diag
            self.singular_to_realspace = self.singular_to_realspace_diag
            self.entropy = self.entropy_pos
        else:
            self.compute_f_J = self.compute_f_J_offdiag
            self.singular_to_realspace = self.singular_to_realspace_offdiag
            self.entropy = self.entropy_posneg
        self.log('Solving...')
        optarr = []
        alpha = 10 ** 6
        self.ustart = np.zeros((self.n_sv))
        converged = False
        conv = 0.
        while conv < 1:
            o = self.maxent_optimization(alpha, self.ustart, use_bayes=True)

            ustart = o.u_opt
            optarr.append(o)
            alpha /= 10.
            conv = o.convergence

        if len(optarr) <= 1:
            raise RuntimeError('Failed to get a prediction for optimal alpha. '
                               + 'Decrease the error or increase alpha_start.')

        bayes_conv = [o.convergence for o in optarr]
        alpharr = [o.alpha for o in optarr]

        # log(conv) is approximately a linear function from log(alpha),
        # based on this we can predict the optimal alpha quite precisely.
        exp_opt = np.log10(alpharr[-2]) \
                 - np.log10(bayes_conv[-2]) * (np.log10(alpharr[-1])
                                               - np.log10(alpharr[-2])) \
                   / (np.log10(bayes_conv[-1]) - np.log10(bayes_conv[-2]))
        alpha_opt = 10 ** exp_opt
        self.log('prediction for optimal alpha: {}, log10(alpha_opt) = {}'.format(alpha_opt, np.log10(alpha_opt)))

        # Starting from the predicted value of alpha, and starting the optimization at the solution for the next-lowest alpha,
        # we find the optimal alpha by newton's root finding method.

        def root_fun(alpha, u0):  # this function is just for the newton root-finding
            res = self.maxent_optimization(alpha, u0, iterfac=100000, use_bayes=True)
            optarr.append(res)
            u0[:] = res.u_opt
            return res.convergence - 1.

        ustart = optarr[-2].u_opt
        alpha_opt = opt.newton(root_fun, alpha_opt, tol=1e-6, args=(ustart,))
        self.log('final optimal alpha: {}, log10(alpha_opt) = {}'.format(alpha_opt, np.log10(alpha_opt)))

        sol = self.maxent_optimization(alpha_opt, ustart, iterfac=250000, use_bayes=True)
        self.alpha_opt = alpha_opt
        self.A_opt = sol.A_opt
        return sol, optarr


    def solve_bryan(self, alphastart=500, alphadiv=1.1, interactive=False):
        """Bryan's method of determining the optimal spectrum.

        Bryan's maxent calculates an average of spectral functions,
        weighted by their Bayesian probability

        Parameters
        ----------
        alphastart : float, default=500
                  Starting value of alpha. This is the largest value of alpha,
                  it is decreased during the calculation.
        alphadiv : float, default=1.1
                  After each optimization, the current alpha is divided by this number.
                  Hence, the number has to be larger than 1.
        interactive : bool, default=False
                  Whether to show a plot of the probability. (Needs matplotlib)

        Returns
        -------
        OptimizationResult
               Contains the weighted average of all results
        list
               List of OptimizationResult objects for all used values of alpha
        """

        if interactive:
            import matplotlib.pyplot as plt

        self.log('Solving...')
        optarr = []
        alpha = alphastart
        self.ustart = np.zeros((self.n_sv))
        maxprob = 0.
        while True:
            o = self.maxent_optimization(alpha, self.ustart, use_bayes=True)
            ustart = o.u_opt
            optarr.append(o)
            alpha /= alphadiv
            prob = o.probability
            # automatically terminate the loop when probability is small again
            if prob > maxprob:
                maxprob = prob
            elif prob < 0.01 * maxprob:
                break

        alpharr = np.array([o.alpha for o in optarr])
        probarr = np.array([o.probability for o in optarr])
        specarr = np.array([o.A_opt for o in optarr])
        probarr /= -np.trapz(probarr, alpharr)  # normalize the probability distribution
        if interactive:
            fig = plt.figure()
            plt.plot(np.log10(alpharr), probarr)
            fig.show()

        # calculate the weighted average spectrum
        A_opt = -np.trapz(specarr * probarr[:, None], alpharr,
                          axis=0)  # need a "-" sign, because alpha goes from large to small.

        sol = OptimizationResult(A_opt=A_opt,
                                 backtransform=self.backtransform(A_opt))

        return sol, optarr

    def solve_chi2kink(self, alpha_start=1e9, alpha_end=1e-3, alpha_div=10., fit_position=2.5, interactive=False, **kwargs):
        """Determine optimal alpha by searching for the kink in log(chi2) (log(alpha))

        We start with an optimization at a large value of alpha (`alpha_start`),
        where we should get only the default model. Then, alpha is decreased
        step-by-step, where :math:`\\alpha_{n+1} = \\alpha_n / \\alpha_{div}`, until the minimal
        value of alpha_end is reached.
        Then, we fit a function
        :math:`\\phi(x; a, b, c, d) = a + b / [1 + exp(-d*(x-c))]`,
        from which the optimal alpha is determined by
        x_opt = c - `fit_position` / d; alpha_opt = 10^x_opt.

        Parameters
        ----------
        alpha_start : float, default=1e9
                  Value of alpha where to start the procedure.
                  This is the largest value of alpha, it is subsequently decreased
                  in the algorithm.
        alpha_end : float, default=1e-3
                  Last (smallest) value of alpha that is considered in the algorithm
        alpha_div : float, default=10.
                  After each optimization, alpha is divided by alpha_div.
                  Thus it has to be larger than 1. The default value of 10 is
                  a good compromise of function and speed. You can take
                  smaller values if you are unsure or want to make fancy plots.
        fit_position : float, default=2.5
                  Control parameter for under/overfitting.
                  In my experience, good values are usually between 2 and 2.5.
                  Smaller values lead to underfitting, which is sometimes desirable.
                  Larger values lead to overfitting, which should be avoided.
        interactive : bool, default=False
                  Decide whether to show a plot of chi2 vs alpha.

        Returns
        -------
        OptimizationResult
               Result of the optimization at "best" alpha value
        list
               List of OptimizationResult objects for all used values of alpha
        """

        if interactive:
            import matplotlib.pyplot as plt

        alpha = alpha_start
        chi = []
        alphas = []
        optarr = []
        ustart = np.zeros((self.n_sv))
        while True:
            try:
                o = self.maxent_optimization(alpha=alpha, ustart=ustart)
                optarr.append(o)
                ustart = o.u_opt
                chi.append(o.chi2)
                alphas.append(alpha)
            except:
                # For small alphas sometimes the optimization fails
                # Usually this happens at values of alpha that
                # are too small anyway, so don't worry.
                print('Optimization at alpha={} failed.'.format(alpha))
            alpha = alpha / alpha_div
            if alpha < alpha_end:
                break

        alphas = np.asarray(alphas)
        chis = np.asarray(chi)

        def fitfun(x, a, b, c, d):
            return a + b / (1. + np.exp(-d * (x - c)))

        try:
            good_numbers = np.isfinite(chis)
            popt, pcov = opt.curve_fit(fitfun,
                                       np.log10(alphas[good_numbers]),
                                       np.log10(chis[good_numbers]),
                                       p0=(0., 5., 2., 0.))
        except ValueError:
            print('Fermi fit failed.')
            if interactive:
                plt.plot(np.log10(alphas), np.log10(chis), marker='s')
                plt.show()
                for o in optarr:
                    plt.plot(o.backtransform)
                plt.plot(self.im_data)
                plt.show()
            return optarr[-1], optarr

        a, b, c, d = popt

        if interactive:
            print('Fit parameters {}'.format(popt))
        if d < 0.:
            raise RuntimeError('Fermi fit temperature negative.')

        a_opt = c - fit_position / d
        alpha_opt = 10. ** a_opt
        self.log('Optimal log alpha {}'.format(a_opt))

        if interactive:
            plt.plot(np.log10(alphas), np.log10(chis), marker='s', label='chi2')
            plt.plot(np.log10(alphas), fitfun(np.log10(alphas), *popt), label='fit2lin')
            #plt.plot(np.log10(alphas), chi2_interp(np.log10(alphas)), label='chi2 fit')

        closest_idx = np.argmin(np.abs(np.log10(alphas) - a_opt))
        ustart = optarr[closest_idx].u_opt
        sol = self.maxent_optimization(alpha_opt, ustart)

        if interactive:
            plt.plot(a_opt, np.log10(sol.chi2), marker='s', color='red', label='opt')
            plt.legend()
            plt.show()

        return sol, optarr

    def error_propagation(self, obs, args):
        """Calculate the deviation for a callable function obs.
        This feature is still experimental"""
        hess = -self.d2chi2
        hess += self.alpha_opt * np.diag(1. / self.A_opt) \
                * self.dw[:, None] * self.dw[None, :]  # possibly, it's 2*alpha
        varmat = np.linalg.inv(hess)  # TODO actually here should be a minus...
        obsval = obs(self.re_axis, args)
        integrand = obsval[:, None] * obsval[None, :] \
                    * varmat * self.dw[:, None] * self.dw[None, :]
        deviation = np.sum(integrand)
        return deviation

    def solve(self, **kwargs):
        """Wrapper function for solve, which calls the chosen
        method of alpha_determination."""
        try:
            alpha_determination = kwargs['alpha_determination']
        except KeyError:
            raise KeyError("No valid alpha determination mode found. Recommended: alpha_determination='chi2kink'")
        if alpha_determination == 'historic':
            return self.solve_historic()
        elif alpha_determination == 'classic':
            return self.solve_classic()
        elif alpha_determination == 'bryan':
            return self.solve_bryan()
        elif alpha_determination == 'chi2kink':
            return self.solve_chi2kink(**kwargs)
        else:
            raise ValueError('Unknown alpha determination mode')


class OptimizationResult(object):
    """Object for holding the result of an optimization.

    This class has no methods except the constructor,
    it is thus essentially a collection of output numbers.

    All member variables have None as default value, different solvers
    override different variables. A_opt is always set, since it
    is the main result of analytic continuation.
    """

    def __init__(self,
                 u_opt=None,
                 A_opt=None,
                 chi2=None,
                 backtransform=None,
                 entropy=None,
                 n_good=None,
                 probability=None,
                 alpha=None,
                 convergence=None,
                 trace=None,
                 Q=None,
                 norm=None,
                 blur_width=None,
                 numerator=None,
                 denominator=None,
                 numerator_function=None,
                 denominator_function=None,
                 check=None,
                 ivcheck=None,
                 g_ret=None):
        self.u_opt = u_opt
        self.A_opt = A_opt
        self.chi2 = chi2
        self.backtransform = backtransform
        self.entropy = entropy
        self.n_good = n_good
        self.probability = probability
        self.alpha = alpha
        self.convergence = convergence
        self.trace = trace
        self.Q = Q
        self.norm = norm
        self.blur_width = blur_width
        self.numerator=numerator
        self.denominator=denominator
        self.numerator_function=numerator_function
        self.denominator_function=denominator_function
        self.check = check
        self.ivcheck = ivcheck
        self.g_ret = g_ret


class NewtonOptimizer(object):
    """Newton root finding."""

    def __init__(self, opt_size, max_hist=1, max_iter=20000, initial_guess=None):
        """
        :param opt_size: number of variables (integer)
        :param max_hist: maximal history for mixing (integer)
        :param max_iter: maximum number of iterations for root finding
        :param initial_guess: initial guess for the root finding.
        """

        if initial_guess is None:
            initial_guess = np.zeros((opt_size))

        self.props = [initial_guess]
        self.res = []
        self.max_hist = max_hist
        self.max_iter = max_iter
        self.opt_size = opt_size
        self.return_object = collections.namedtuple("NewtonResult", ['x', 'nfev'])

    def iteration_function(self, proposal, function_vector, jacobian_matrix):
        """The function, whose fixed point we are searching.

        This function generates the iteration procedure in the Newton root finding.
        Basically, it computes the "result" from the "proposal".
        It has the form result = proposal + increment, but
        since the increment may be large, we apply a reduction of step width in such cases.
        """

        try:
            increment = -np.dot(np.linalg.pinv(jacobian_matrix), function_vector)
        except np.linalg.LinAlgError:
            print('LinAlgError in Newton Solver, setting increment to zero')
            increment = np.zeros_like(proposal)
        step_reduction = 1.
        significance_limit = 1e-4
        if np.any(np.abs(proposal) > significance_limit):
            ratio = np.abs(increment / proposal)
            max_ratio = np.amax(ratio[np.abs(proposal) > significance_limit])
            if max_ratio > 1.:
                step_reduction = 1. / max_ratio
        result = proposal + step_reduction * increment
        return result

    def __call__(self, function_and_jacobian, alpha):
        """Main function of Newton optimization.

        This function implements the self-consistent iteration of the root finding.

        Returns
        -------
        collections.namedtuple
               sol.x is the result
               sol.nfev is the number of function evaluations
        """
        f, J = function_and_jacobian(self.props[0], alpha)
        initial_result = self.iteration_function(self.props[0], f, J)
        self.res.append(initial_result)

        counter = 0
        converged = False
        while not converged:
            prop = self.get_proposal()
            f, J = function_and_jacobian(prop, alpha)
            result = self.iteration_function(prop, f, J)
            self.props.append(prop)
            self.res.append(result)
            converged = (counter > self.max_iter or np.max(np.abs(result - prop)) < 1e-4)
            counter += 1
            if np.any(np.isnan(result)):
                raise RuntimeWarning('Function returned NaN.')
        if counter > self.max_iter:
            raise RuntimeWarning('Failed to get optimization result in {} iterations'.format(self.max_iter))

        self.return_object.x = result
        self.return_object.nfev = counter
        return self.return_object

    def get_proposal(self, mixing=0.5):
        """Propose a new solution by linear mixing."""

        n_iter = len(self.props)
        history = min(self.max_hist, n_iter) - 1

        new_proposal = self.props[n_iter - 1]
        f_i = self.res[n_iter - 1] - self.props[n_iter - 1]
        update = mixing * f_i
        return new_proposal + update

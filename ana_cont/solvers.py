import numpy as np
import scipy.optimize as opt
from . import pade

class AnalyticContinuationSolver(object):
    pass

# class for return value of maxent_optimization
class OptimizationResult:
    pass

class PadeSolver(AnalyticContinuationSolver):
    def __init__(self, im_axis, re_axis, im_data):
        self.im_axis = im_axis
        self.re_axis = re_axis
        self.im_data = im_data

        # Compute the Pade coefficients
        self.a = pade.compute_coefficients(1j*self.im_axis, self.im_data)

    def check(self, show_plot=False):
        # As a check, look if Pade approximant smoothly interpolates the original data
        self.ivcheck = np.linspace(0, 2*np.max(self.im_axis), num=500)
        self.check = pade.C(self.ivcheck*1j, 1j*self.im_axis,
                            self.im_data, self.a)

    def solve(self,show_plot=False):
        # Compute the Pade approximation on the real axis
        self.result = pade.C(self.re_axis, 1j*self.im_axis,
                             self.im_data, self.a)


        return self.result



class MaxentSolverSVD(AnalyticContinuationSolver):
    def __init__(self, im_axis, re_axis, im_data,
                 kernel_mode='', model=None, stdev=None,
                 beta=None, **kwargs):
        self.kernel_mode = kernel_mode
        self.im_axis = im_axis
        self.re_axis = re_axis
        self.im_data = im_data
        self.nw = self.re_axis.shape[0]
        self.wmin = self.re_axis[0]
        self.wmax = self.re_axis[-1]
        self.dw = np.diff(
            np.concatenate(([self.wmin],
                            (self.re_axis[1:] + self.re_axis[:-1]) / 2.,
                            [self.wmax])))
        self.model = model  # the model should be normalized by the user himself

        if self.kernel_mode == 'freq_bosonic':
            self.var = stdev ** 2
            self.E = 1. / self.var
            self.niw = self.im_axis.shape[0]
            self.kernel = (self.re_axis ** 2)[None, :] \
                          / ((self.re_axis ** 2)[None, :]
                             + (self.im_axis ** 2)[:, None])
            self.kernel[0, 0] = 1.  # analytically with de l'Hospital
        elif self.kernel_mode == 'time_bosonic':
            self.var = stdev ** 2
            self.E = 1. / self.var
            self.niw = self.im_axis.shape[0]
            self.kernel = 0.5 * self.re_axis[None, :] * (
                np.exp(-self.re_axis[None, :] * self.im_axis[:, None])
                + np.exp(-self.re_axis[None, :] * (1. - self.im_axis[:, None]))) / (
                              1. - np.exp(-self.re_axis[None, :]))
            self.kernel[:, 0] = 1.  # analytically with de l'Hospital
        elif self.kernel_mode == 'freq_fermionic':
            self.var = np.concatenate((stdev ** 2, stdev ** 2))
            self.E = 1. / self.var
            self.niw = 2 * self.im_axis.shape[0]
            self.kernel = np.zeros((self.niw, self.nw))  # fermionic Matsubara GF is complex
            self.kernel[:self.niw / 2, :] = -self.re_axis[None, :] / (
            (self.re_axis ** 2)[None, :] + (self.im_axis ** 2)[:, None])
            self.kernel[self.niw / 2:, :] = -self.im_axis[:, None] / (
            (self.re_axis ** 2)[None, :] + (self.im_axis ** 2)[:, None])
        elif self.kernel_mode == 'time_fermionic':
            self.var = stdev ** 2
            self.E = 1. / self.var
            self.niw = self.im_axis.shape[0]
            self.kernel = np.exp(-self.im_axis[:, None] * self.re_axis[None, :]) \
                          / (1. + np.exp(-self.re_axis[None, :]))
        elif self.kernel_mode == 'freq_fermionic_phsym':  # in this case, the data must be purely real (the imaginary part!)
            print('Warning: phsym kernels do not give good results in this implementation. ')
            self.var = stdev ** 2
            self.E = 1. / self.var
            self.niw = self.im_axis.shape[0]
            self.kernel = -2. * self.im_axis[:, None] \
                          / ((self.im_axis ** 2)[:, None] + (self.re_axis ** 2)[None, :])
        elif self.kernel_mode == 'time_fermionic_phsym':
            print('Warning: phsym kernels do not give good results in this implementation. ')
            self.var = stdev ** 2
            self.E = 1. / self.var
            self.niw = self.im_axis.shape[0]
            self.kernel = (np.cosh(self.im_axis[:, None] * self.re_axis[None, :])
                           + np.cosh((1. - self.im_axis[:, None]) * self.re_axis[None, :])) / (
                          1. + np.cosh(self.re_axis[None, :]))
        else:
            print('Unknown kernel')
            sys.exit()

        U, S, Vt = np.linalg.svd(self.kernel, full_matrices=False)

        self.n_sv = np.arange(min(self.nw, self.niw))[S > 1e-10][-1]  # number of singular values larger than 1e-10

        self.U_svd = np.array(U[:, :self.n_sv], dtype=np.float64, order='C')
        self.V_svd = np.array(Vt[:self.n_sv, :].T, dtype=np.float64, order='C')  # numpy.svd returns V.T
        self.Xi_svd = S[:self.n_sv]

        print('spectral points:', self.nw)
        print('data points on imaginary axis:', self.niw)
        print('significant singular values:', self.n_sv)
        print('U', self.U_svd.shape)
        print('V', self.V_svd.shape)
        print('Xi', self.Xi_svd.shape)

        # =============================================================================================
        # First, precompute as much as possible
        # The precomputation of W2 is done in C, this saves a lot of time!
        # The other precomputations need less loop, can stay in python for the moment.
        # =============================================================================================
        print('Precomputation of coefficient matrices')

        # allocate space
        self.W2 = np.zeros((self.n_sv, self.nw), order='C', dtype=np.float64)
        self.W3 = np.zeros((self.n_sv, self.n_sv, self.nw))
        self.d2chi2 = np.zeros((self.nw, self.nw))
        self.Evi = np.zeros((self.n_sv))

        # precompute matrices W_ml (W2), W_mil (W3)
        self.W2 = np.einsum('k,km,m,kn,n,ln,l,l->ml', self.E, self.U_svd, self.Xi_svd, self.U_svd, self.Xi_svd,
                            self.V_svd, self.dw, self.model)
        self.W3 = self.W2[:, None, :] * (self.V_svd[None, :, :]).transpose((0, 2, 1))

        # precompute the evidence vector Evi_m
        self.Evi = np.einsum('m,km,k,k->m', self.Xi_svd, self.U_svd, self.E, self.im_data)

        # precompute curvature of likelihood function
        self.d2chi2 = np.einsum('i,j,ki,kj,k->ij', self.dw, self.dw, self.kernel, self.kernel, self.E)

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



    # define derivative f_m(u) = SVD(dQ/dA)_m [we search for zeros of f(u)]
    # and the Jacobian matrix of f, J_mi=df_m/du_i
    # TODO this is the main function call, it must be optimized as much as possible!
    def compute_f_J(self, u, alpha):
        v = np.dot(self.V_svd, u)
        w = np.exp(v)
        f = alpha * u + np.dot(self.W2, w) - self.Evi
        J = alpha * np.eye(self.n_sv) + np.dot(self.W3, w)
        return f, J

    # =============================================================================================
    # Some auxiliary functions
    # =============================================================================================

    # transform the singular space vector u into real space (spectral function)
    def singular2realspace(self, u):
        return self.model * np.exp(np.dot(self.V_svd, u))

    # backtransformation from real to imaginary axis
    def backtransform(self, A):
        return np.dot(self.kernel, A * self.dw)

    # compute the log-likelihood function of A
    def chi2(self, A):
        return np.sum(self.E * (self.im_data - np.dot(self.kernel, A * self.dw)) ** 2)

    def entropy(self, A, u):
        return np.trapz(A - self.model - A * np.dot(self.V_svd, u), self.re_axis)

    # Bayesian convergence criterion for classic maxent (maximum of probablility distribution)
    def bayes_conv(self, A, entr, alpha):
        LambdaMatrix = np.sqrt(A / self.dw)[:, None] * self.d2chi2 * np.sqrt(A / self.dw)[None, :]
        lam = np.linalg.eigvalsh(LambdaMatrix)
        ng = -2. * alpha * entr
        tr = np.sum(lam / (alpha + lam))
        conv = tr / ng
        return ng, tr, conv

    # Bayesian a-posteriori probability for alpha after optimization of A
    def posterior_probability(self, A, alpha, entr, chisq):
        lambda_matrix = np.sqrt(A / self.dw)[:, None] * self.d2chi2 * np.sqrt(A / self.dw)[None, :]
        lam = np.linalg.eigvalsh(lambda_matrix)
        eig_sum = np.sum(np.log(alpha / (alpha + lam)))
        log_prob = alpha * entr - 0.5 * chisq + np.log(alpha) + 0.5 * eig_sum
        return np.exp(log_prob)

    # optimization of maxent functional for a given value of alpha
    def maxent_optimization(self, alpha, ustart, iterfac=10000000):
        sol = opt.root(self.compute_f_J,  # function returning function value f and jacobian J (we search root of f)
                       ustart,  # sensible starting point
                       method='lm',  # levenberg-marquart method
                       jac=True,  # already self.compute_f_J returns the jacobian (slightly more efficient in this case)
                       options={'maxiter': iterfac * self.n_sv,  # max number of lm steps
                                'factor': 100.,  # scale for initial stepwidth of lm (?)
                                'diag': np.exp(np.arange(self.n_sv))},
                       # scale for values to find (assume that they decay exponentially)
                       args=(alpha))  # additional argument for self.compute_f_J
        u_opt = sol.x
        A_opt = self.singular2realspace(sol.x)
        entr = self.entropy(A_opt, u_opt)
        chisq = self.chi2(A_opt)
        ng, tr, conv = self.bayes_conv(A_opt, entr, alpha)
        norm = np.trapz(A_opt, self.re_axis)
        print('log10(alpha)={:6.4f}\tchi2={:5.4e}\tS={:5.4e}\ttr={:5.4f}\tconv={:1.3},\tnfev={},\tnorm={}'.format(
            np.log10(alpha), chisq, entr, tr, conv, sol.nfev, norm))
        result = OptimizationResult()
        result.u_opt = u_opt
        result.A_opt = A_opt
        result.alpha = alpha
        result.entropy = entr
        result.backtransform = self.backtransform(A_opt)
        result.n_good = ng
        result.trace = tr
        result.convergence = conv
        result.norm = norm
        result.probability = self.posterior_probability(A_opt, alpha, entr, chisq)
        result.Q = alpha * entr - 0.5 * chisq
        return result

    # =============================================================================================
    # Now actually solve the problem!
    # =============================================================================================


    # classic maxent uses Bayes statistics to approximately determine
    # the most probable value of alpha
    # We start at a large value of alpha, where the optimization yields basically the default model,
    # therefore u_opt is only a few steps away from ustart=0 (=default model)
    # Then we gradually decrease alpha, step by step moving away from the default model towards the evidence.
    # Using u_opt as ustart for the next (smaller) alpha brings a great speedup into this procedure.
    def solve_classic(self):  # classic maxent
        print('Solving...')
        optarr = []
        alpha = 10 ** 5
        self.ustart = np.zeros((self.n_sv))
        converged = False
        conv = 0.
        while conv < 1:
            o = self.maxent_optimization(alpha, self.ustart)

            ustart = o.u_opt
            optarr.append(o)
            alpha /= 10.
            conv = o.convergence

        bayes_conv = [o.convergence for o in optarr]
        alpharr = [o.alpha for o in optarr]

        # log(conv) is approximately a linear function from log(alpha),
        # based on this we can predict the optimal alpha quite precisely.
        expOpt = np.log10(alpharr[-2]) \
                 - np.log10(bayes_conv[-2]) * (np.log10(alpharr[-1])
                                               - np.log10(alpharr[-2])) \
                   / (np.log10(bayes_conv[-1]) - np.log10(bayes_conv[-2]))
        alphaOpt = 10 ** expOpt
        print('prediction for optimal alpha:', alphaOpt, 'log10(alphaOpt)=', np.log10(alphaOpt))

        # Starting from the predicted value of alpha, and starting the optimization at the solution for the next-lowest alpha,
        # we find the optimal alpha by newton's root finding method.

        def root_fun(alpha, u0):  # this function is just for the newton root-finding
            res = self.maxent_optimization(alpha, u0, iterfac=100000)
            optarr.append(res)
            u0[:] = res.u_opt
            return res.convergence - 1.

        ustart = optarr[-2].u_opt
        alpha_opt = opt.newton(root_fun, alphaOpt, tol=1e-6, args=(ustart,))
        print('final optimal alpha:', alpha_opt, 'log10(alpha_opt)=', np.log10(alpha_opt))

        sol = self.maxent_optimization(alpha_opt, ustart, iterfac=250000)
        self.alpha_opt = alpha_opt
        self.A_opt = sol.A_opt
        return sol, optarr

    # Bryan's maxent calculates an average of spectral functions,
    # weighted by their Bayesian probability
    def solve_bryan(self, alphastart=500, alphadiv=1.1):
        print('Solving...')
        optarr = []
        alpha = alphastart
        self.ustart = np.zeros((self.n_sv))
        maxprob = 0.
        while True:
            o = self.maxent_optimization(alpha, self.ustart)
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
        fig = plt.figure()
        plt.plot(np.log10(alpharr), probarr)
        fig.show()

        # calculate the weighted average spectrum
        A_opt = -np.trapz(specarr * probarr[:, None], alpharr,
                          axis=0)  # need a "-" sign, because alpha goes from large to small.

        sol = OptimizationResult()
        sol.A_opt = A_opt
        sol.backtransform = self.backtransform(A_opt)
        return sol, optarr

    # calculate the deviation for a callable function obs
    # this feature is still experimental
    def error_propagation(self, obs, args):
        hess = -self.d2chi2
        hess += self.alpha_opt * np.diag(1. / self.A_opt) \
                * self.dw[:, None] * self.dw[None, :]  # possibly, it's 2*alpha
        varmat = np.linalg.inv(hess)  # TODO actually here should be a minus...
        obsval = obs(self.re_axis, args)
        integrand = obsval[:, None] * obsval[None, :] \
                    * varmat * self.dw[:, None] * self.dw[None, :]
        deviation = np.sum(integrand)
        return deviation

    # switch for different types of alpha selection
    def solve(self, alpha_determination='classic'):

        if alpha_determination == 'classic':
            return self.solve_classic()
        elif alpha_determination == 'bryan':
            return self.solve_bryan()


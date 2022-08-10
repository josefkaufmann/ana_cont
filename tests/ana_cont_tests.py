import unittest
import numpy as np
import ana_cont.continuation as cont
import ana_cont.kernels as kernels


class TestGreensFunction(unittest.TestCase):
    def setUp(self):
        w = np.linspace(-10., 10., endpoint=True, num=21)  # use larger num for real cases!
        spec = np.exp(-((w-1.)/0.5)**2)
        spec /= np.trapz(spec, w)
        self.greens_function = cont.GreensFunction(spectrum=spec, wgrid=w, kind='fermionic')
        self.complex_greens_function = np.array([-9.09358619e-02-1.91865208e-210j, -1.00035695e-01-5.80407090e-174j,
                                               -1.11160191e-01-5.88997340e-141j, -1.25070115e-01-2.00510954e-111j,
                                               -1.42962315e-01-2.28984991e-085j, -1.66834942e-01-8.77243332e-063j,
                                               -2.00294482e-01-1.12739805e-043j, -2.50588965e-01-4.86047404e-028j,
                                               -3.34805759e-01-7.02949492e-016j, -5.05889391e-01-3.41046628e-007j,
                                               -9.73497083e-01-5.55069727e-002j, -8.76179213e-19-3.03057803e+000j,
                                                9.73497083e-01-5.55069727e-002j,  5.05889391e-01-3.41046628e-007j,
                                                3.34805759e-01-7.02949492e-016j,  2.50588965e-01-4.86047404e-028j,
                                                2.00294482e-01-1.12739805e-043j,  1.66834942e-01-8.77243332e-063j,
                                                1.42962315e-01-2.28984991e-085j,  1.25070115e-01-2.00510954e-111j,
                                                1.11160191e-01-5.88997340e-141j])

    def test_kkt(self):
        self.assertTrue(np.allclose(self.greens_function.kkt(), self.complex_greens_function))


class TestKernels(unittest.TestCase):
    def setUp(self):
        self.w_real = np.linspace(0., 1., num=3, endpoint=True)
        self.iw_ferm = 2. * np.arange(3) + 1.
        self.iw_bos = 2 * np.arange(3)
        self.imag_time = np.linspace(0., 1., num=3, endpoint=False)
        self.kernel_freq_fermionic = kernels.Kernel(im_axis=self.iw_ferm, re_axis=self.w_real, kind='freq_fermionic')
        self.kernel_freq_bosonic = kernels.Kernel(im_axis=self.iw_bos, re_axis=self.w_real, kind='freq_bosonic')
        self.kernel_time_fermionic = kernels.Kernel(im_axis=self.imag_time, re_axis=self.w_real, kind='time_fermionic')
        self.kernel_time_bosonic = kernels.Kernel(im_axis=self.imag_time, re_axis=self.w_real, kind='time_bosonic')

    def test_freq_fermionic(self):
        correct_matrix = np.array([[0.-1.j, -0.4-0.8j, -0.5-0.5j],
                                   [0.-0.33333333j, -0.05405405-0.32432432j, -0.1-0.3j],
                                   [0.-0.2j, -0.01980198-0.1980198j, -0.03846154-0.19230769j]])
        self.assertTrue(np.allclose(self.kernel_freq_fermionic.matrix, correct_matrix))

    def test_freq_bosonic(self):
        correct_matrix = np.array([[1., 1., 1.],
                                   [0., 0.05882353, 0.2],
                                   [0., 0.01538462, 0.05882353]])
        self.assertTrue(np.allclose(self.kernel_freq_bosonic.matrix, correct_matrix))

    def test_time_fermionic(self):
        correct_matrix = np.array([[0.5, 0.62245933, 0.73105858],
                                   [0.5, 0.52690045, 0.52382636],
                                   [0.5, 0.4460116, 0.37533799]])
        self.assertTrue(np.allclose(self.kernel_time_fermionic.matrix, correct_matrix))

    def test_time_bosonic(self):
        correct_matrix = np.array([[1., 1.02074704, 1.08197671],
                                   [1., 0.9930971, 0.97287488],
                                   [1., 0.9930971, 0.97287488]])
        self.assertTrue(np.allclose(self.kernel_time_bosonic.matrix, correct_matrix))

class TestMaxentFermionic(unittest.TestCase):
    def setUp(self):
        # real-frequency grid and example spectrum
        w = np.linspace(-10., 10., num=501, endpoint=True)
        spec = 0.4 * np.exp(-0.5 * (w - 1.8) ** 2) + 0.6 * np.exp(-0.5 * (w + 1.8) ** 2)
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
        self.sol, _ = probl.solve(method='maxent_svd', alpha_determination='chi2kink', optimizer='newton',
                             model=model, stdev=err, interactive=False, alpha_start=1e12, alpha_end=1e-2,
                             preblur=True, blur_width=0.5, verbose=False)

        self.A_opt_reference = np.load("tests/A_opt_maxent_fermionic.npy")
        self.backtransform_reference = np.load("tests/backtransform_maxent_fermionic.npy")
        self.chi2_reference = np.load("tests/chi2_maxent_fermionic.npy")

    def test_continuation(self):
        self.assertTrue(np.allclose(self.sol.A_opt, self.A_opt_reference))

    def test_backtransform(self):
        self.assertTrue(np.allclose(self.backtransform_reference, self.sol.backtransform))

    def test_chi2(self):
        self.assertAlmostEqual(self.chi2_reference, self.sol.chi2, places=4)


class TestPade(unittest.TestCase):
    def setUp(self):
        # real-frequency grid and example spectrum
        w = np.linspace(-10., 10., num=501, endpoint=True)
        spec = 0.4 * np.exp(-0.5 * (w - 1.8) ** 2) + 0.6 * np.exp(-0.5 * (w + 1.8) ** 2)
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
        self.sol = probl.solve(method='pade')
        check_axis = np.linspace(0., 1.25 * iw[mats_ind[-1]], num=500)
        self.check = probl.solver.check(im_axis_fine=check_axis)

        self.A_opt_reference = np.load("tests/A_opt_pade.npy")
        self.check_reference = np.load("tests/check_pade.npy")

    def test_continuation(self):
        self.assertTrue(np.allclose(self.sol.A_opt, self.A_opt_reference))

    def test_check(self):
        self.assertTrue(np.allclose(self.check, self.check_reference))

class TestMaxentBosonic(unittest.TestCase):
    def setUp(self):
        w_real = np.linspace(0., 5., num=2001, endpoint=True)
        spec_real = np.exp(-(w_real) ** 2 / (2. * 0.2 ** 2))
        spec_real += 0.3 * np.exp(-(w_real - 1.5) ** 2 / (2. * 0.8 ** 2))
        spec_real += 0.3 * np.exp(-(w_real + 1.5) ** 2 / (2. * 0.8 ** 2))  # must be symmetric around 0!
        spec_real /= np.trapz(spec_real, w_real)  # normalization

        beta = 10.
        iw = 2. * np.pi / beta * np.arange(10)

        noise_amplitude = 1e-4  # create gaussian noise
        rng = np.random.RandomState(1234)
        noise = rng.normal(0., noise_amplitude, iw.shape[0])

        with np.errstate(invalid="ignore"):
            kernel = (w_real ** 2)[None, :] / ((iw ** 2)[:, None] + (w_real ** 2)[None, :])
        kernel[0, 0] = 1.
        gf_bos = np.trapz(kernel * spec_real[None, :], w_real, axis=1) + noise
        norm = gf_bos[0]
        gf_bos /= norm

        w = np.linspace(0., 5., num=501, endpoint=True)
        probl = cont.AnalyticContinuationProblem(im_axis=iw, re_axis=w,
                                                 im_data=gf_bos, kernel_mode='freq_bosonic')

        err = np.ones_like(iw) * noise_amplitude / norm
        model = np.ones_like(w)
        model /= np.trapz(model, w)
        self.sol, _ = probl.solve(method='maxent_svd',
                             alpha_determination='chi2kink',
                             optimizer='newton',
                             stdev=err, model=model,
                             interactive=False, verbose=False)

        self.A_opt_reference = np.load("tests/A_opt_maxent_bosonic.npy")
        self.backtransform_reference = np.load("tests/backtransform_maxent_bosonic.npy")
        self.chi2_reference = np.load("tests/chi2_maxent_bosonic.npy")

    def test_continuation(self):
        self.assertTrue(np.allclose(self.sol.A_opt, self.A_opt_reference))

    def test_backtransform(self):
        self.assertTrue(np.allclose(self.backtransform_reference, self.sol.backtransform))

    def test_chi2(self):
        self.assertAlmostEqual(self.chi2_reference, self.sol.chi2, places=4)



if __name__ == '__main__':
    unittest.main()
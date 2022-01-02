import unittest
import numpy as np
import ana_cont.continuation as cont


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

        self.A_opt_reference = np.load("A_opt_maxent_fermionic.npy")
        self.backtransform_reference = np.load("backtransform_maxent_fermionic.npy")
        self.chi2_reference = np.load("chi2_maxent_fermionic.npy")

    def test_continuation(self):
        self.assertTrue(np.allclose(self.sol.A_opt, self.A_opt_reference))

    def test_backtransform(self):
        self.assertTrue(np.allclose(self.backtransform_reference, self.sol.backtransform))

    def test_chi2(self):
        self.assertEqual(self.chi2_reference, self.sol.chi2)


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

        self.A_opt_reference = np.load("A_opt_pade.npy")
        self.check_reference = np.load("check_pade.npy")

    def test_continuation(self):
        self.assertTrue(np.allclose(self.sol.A_opt, self.A_opt_reference))

    def test_check(self):
        self.assertTrue(np.allclose(self.check, self.check_reference))

if __name__ == '__main__':
    unittest.main()
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




if __name__ == '__main__':
    unittest.main()
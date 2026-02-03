import numpy as np
import pytest
import ana_cont.solvers


def test_fit():
    log_alphas = np.arange(-3, 6).astype(float)
    a, b, c, d = 1.0, 3.0, 0.5, 1.2
    log_chis = a + b / (1 + np.exp(-d * (log_alphas - c)))
    fit_position = 2.5

    alpha_opt = ana_cont.solvers.opt_alpha_from_sigmoid_fit(
        10**log_alphas, 10**log_chis, fit_position
    )
    np.testing.assert_almost_equal(
        float(alpha_opt), 
        10**(c - fit_position / d), 
        decimal=7
    )

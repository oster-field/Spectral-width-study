"""Finding parameters for two different spectra so that their widths coincide"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


def f(x, x0):  # My spectrum with power = 6
    return 1 / (1 + (x - x0) ** 6)


def g(x, x0, b):  # Gaussian spectrum
    return np.exp(-b * (x - x0) ** 2)


def moments_f(x0):  # Spectral moments for them
    m0, _ = quad(lambda x: f(x, x0), 0, np.inf)
    m2, _ = quad(lambda x: x ** 2 * f(x, x0), 0, np.inf)
    m4, _ = quad(lambda x: x ** 4 * f(x, x0), 0, np.inf)
    return m0, m2, m4


def moments_g(x0, b):
    m0, _ = quad(lambda x: g(x, x0, b), 0, np.inf)
    m2, _ = quad(lambda x: x ** 2 * g(x, x0, b), 0, np.inf)
    m4, _ = quad(lambda x: x ** 4 * g(x, x0, b), 0, np.inf)
    return m0, m2, m4


def compute_ratio(m0, m2, m4):  # Width parameter
    return np.sqrt(1 - m2 ** 2 / (m0 * m4))


def objective(params):
    x0_f, x0_g, b_g = params
    m0_f, m2_f, m4_f = moments_f(x0_f)
    ratio_f = compute_ratio(m0_f, m2_f, m4_f)
    m0_g, m2_g, m4_g = moments_g(x0_g, b_g)
    ratio_g = compute_ratio(m0_g, m2_g, m4_g)
    return (ratio_f - ratio_g) ** 2


initial_guess = [1.0, 1.0, 1.0]
result = minimize(objective, initial_guess, bounds=[(0, None), (0, None), (0, None)])
if result.success:
    x0_f_opt, x0_g_opt, b_g_opt = result.x
    print(f"x0 for my spectrum: {x0_f_opt}")
    print(f"x0 for gaussian: {x0_g_opt}")
    print(f"b for gaussian: {b_g_opt}")
    m0_f_opt, m2_f_opt, m4_f_opt = moments_f(x0_f_opt)
    m0_g_opt, m2_g_opt, m4_g_opt = moments_g(x0_g_opt, b_g_opt)
    ratio_f_opt = compute_ratio(m0_f_opt, m2_f_opt, m4_f_opt)
    ratio_g_opt = compute_ratio(m0_g_opt, m2_g_opt, m4_g_opt)

    print(f"Width my spectrum: {ratio_f_opt}")
    print(f"Width gaussian: {ratio_g_opt}")
else:
    print("Fail.")

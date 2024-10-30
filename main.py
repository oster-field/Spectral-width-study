"""File with a class for constructing a distribution function and other auxiliary functions."""
import numpy as np
import random
from PyAstronomy import pyaC
from scipy.special import erf
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class WaveFieldSimulation:
    def __init__(self, num_realizations=200, max_w=900, num_harmonics=2 ** 13,
                 spectrum_w0=17.5, power_spectrum=6, a=1, b=5, kc=1e5, k=1e-5,
                 ampl_or_extr='amplitude', name_of_spectrum='Parabolic', showcase=True, save_file=True):
        # Initialize parameters
        self.num_realizations = num_realizations  # Number of realizations
        self.max_w = max_w  # The last frequency in spectrum
        self.num_harmonics = num_harmonics  # Number of harmonics
        self.spectrum_w0 = spectrum_w0  # x-coord of peak (for all spectra)
        self.power_spectrum = power_spectrum  # param. of my spectrum
        self.a = a  # param. of gaussian
        self.b = b  # param. of gaussian
        self.kc = kc  # param. of parabolic
        self.k = k  # param. of parabolic
        self.ampl_or_extr = ampl_or_extr  # CDF of amplitudes or extrema or non-global extrema
        self.name_of_spectrum = name_of_spectrum  # Name of spectrum
        self.showcase = showcase  # Show spectrum and one realization or not
        self.save_file = save_file  # Save file or not

    def spectrum(self, w):
        if self.name_of_spectrum == 'My spectrum':
            return 1 / (1 + (w - self.spectrum_w0) ** self.power_spectrum)
        elif self.name_of_spectrum == 'Gaussian':
            return self.a * np.exp(-self.b * (w - self.spectrum_w0) ** 2)
        elif self.name_of_spectrum == 'Parabolic':
            q = ((self.k * self.kc) / (self.k + 2 * np.sqrt(self.k * self.kc) + self.kc)) ** (1 / 3)
            w0 = np.sqrt(q / self.kc) + self.spectrum_w0
            if w0 - np.sqrt(q / self.kc) <= w <= w0:
                return -self.kc * (w - w0) ** 2 + q
            elif w0 < w <= w0 + np.sqrt(q / self.k):
                return -self.k * (w - w0) ** 2 + q
            else:
                return 0

    def run_wave_field(self, i):
        t, y, sx, sy, dw, dt = wave_field(self.max_w, self.num_harmonics, self.spectrum)
        if i == 0 and self.showcase:
            plt.plot(sx, sy, color='red', linewidth=2, marker='.')
            plt.show()
            plt.plot(t, y, color='b', alpha=0.9, marker='.')
            plt.show()
        max_array = zero_crossing(t, y, self.ampl_or_extr)
        significant_ampl = 2 * np.sqrt(np.trapz(sy, dx=dw))
        return max_array / significant_ampl

    def run_simulation(self):
        CDF_points = np.array([])
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.run_wave_field, range(self.num_realizations)),
                                total=self.num_realizations, colour='green'))
        for result in results:
            CDF_points = np.append(CDF_points, result)
        CDF_y = np.linspace(1, 0, len(CDF_points), endpoint=False)
        CDF_x = np.sort(CDF_points)

        if not os.path.isdir(self.name_of_spectrum):
            os.mkdir(self.name_of_spectrum)
        _, _, s_x, s_y, dx, _ = wave_field(self.max_w, self.num_harmonics, self.spectrum)
        m0 = np.trapz(s_y, dx=dx)
        m2 = np.trapz(s_x ** 2 * s_y, dx=dx)
        m4 = np.trapz(s_x ** 4 * s_y, dx=dx)
        epsilon = np.sqrt(1 - (m2 ** 2) / (m0 * m4))

        save = {'x': CDF_x, 'y': CDF_y, 'width': epsilon}
        if self.save_file:
            np.save(f'{self.name_of_spectrum}/R{self.num_realizations} MW{self.max_w} NH{self.num_harmonics}.npy', save)
            print(f'Saved, width = {epsilon}, file name: {self.name_of_spectrum}/R{self.num_realizations} '
                  f'MW{self.max_w} NH{self.num_harmonics}')
        return save


def wave_field(last_w, summ_num, spectrum, **kwargs):  # Making a wave field
    dt = 2 * np.pi / last_w  # Time step
    dw = last_w / summ_num  # Frequency step
    len_rec = dt * summ_num  # Total length of realization
    w = 0
    t = np.arange(0, len_rec, dt)
    eta, s_eta, s_w = 0, np.arange(0), np.arange(0)
    for _ in range(0, summ_num):
        w += dw
        v = random.uniform(0, 2 * np.pi)  # Random phase (uniformly distributed)
        s = spectrum(w, **kwargs)  # Custom spectrum
        s_eta = np.append(s_eta, s)
        s_w = np.append(s_w, w)
        eta += (np.sqrt(2 * dw * s)) * (np.cos(w * t + v))  # Summ of monochromatic waves
    return t, eta, s_w, s_eta, dw, dt


def zero_crossing(t, y, amplitude_extrema):  # Positive amplitudes or local extrema from generated wave field
    array = np.arange(0)
    tc, ti = pyaC.zerocross1d(t, y, getIndices=True)
    t_new = np.sort(np.append(t, tc))
    for n in range(1, len(t_new + 1)):
        if t_new[n] in tc:
            tzm1 = np.where(t_new == t_new[n - 1])[0]
            yzm1 = np.where(y == y[tzm1])[0]
            y = np.insert(y, yzm1 + 1, [0])  # Zero crossings are marked by zero points
    q = np.arange(0)
    for j in y:
        if j == 0:
            q = np.append(q, 0)
            if np.sum(q) > 0:
                if amplitude_extrema == 'amplitude':
                    array = np.append(array, np.max(q))
                elif amplitude_extrema == 'extrema' or amplitude_extrema == 'non-global':
                    dq_1 = np.diff(q)
                    dq_2 = np.diff(q, n=2)
                    for i in range(1, len(dq_2)):
                        if dq_1[i - 1] > 0 >= dq_1[i] and dq_2[i - 1] < 0 < q[i]:
                            if amplitude_extrema == 'extrema':
                                array = np.append(array, q[i])
                            elif amplitude_extrema == 'non-global':
                                if q[i] != np.max(q):
                                    array = np.append(array, q[i])
            q = np.arange(0)
        q = np.append(q, j)
    return array


def spectral_width(spectrum):  # Spectral width parameter proposed by L.Higgins
    w = np.arange(0, 1000, 0.0001)
    s = spectrum(w)
    m0 = np.trapz(s, dx=0.0001)
    m2 = np.trapz((w ** 2) * s, dx=0.0001)
    m4 = np.trapz((w ** 4) * s, dx=0.0001)
    return np.sqrt(1 - (m2 ** 2) / (m0 * m4))


def formula(a, p):  # My formula for CDF of amplitudes
    term1 = ((np.sqrt(2) * (1 + p ** 4)) / p) * a  # Not sure about this term
    term2 = np.exp(-2 * a ** 2)
    term3 = ((np.sqrt(2) * np.sqrt(1 - p ** 2)) / p) * a
    return 0.5 * (1 - erf(term1) + term2 * (1 + erf(term3)))


def f1(x, p):
    return (np.exp(-2 * x**2) / 2) * (1 + erf((x * np.sqrt(2 - 2 * p**2))/p))


def additional_function(x, psi):
    return 0.5 * (1 - erf(x * psi))

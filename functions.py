import numpy as np
import random
from PyAstronomy import pyaC
from scipy.special import erf


def wave_field(last_w, summ_num, spectrum, **kwargs):  # Making a wave field
    dt = 2 * np.pi / last_w  # Time step
    dw = last_w / summ_num  # Frequency step
    len_rec = dt * summ_num  # Total length of realization
    w = 0
    t = np.arange(0, len_rec, dt)
    eta, s_eta, s_w = 0, np.arange(0), np.arange(0)
    for _ in range(0, summ_num):
        w = w + dw
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
                elif amplitude_extrema == 'extrema':
                    dq_1 = np.diff(q)
                    dq_2 = np.diff(q, n=2)
                    for i in range(1, len(dq_2)):
                        if dq_1[i - 1] > 0 >= dq_1[i] and dq_2[i - 1] < 0 < q[i]:
                            array = np.append(array, q[i])
            q = np.arange(0)
        q = np.append(q, j)
    return array


def spectral_width(spectrum, **kwargs):  # Spectral width parameter proposed by L.Higgins
    w = np.arange(0, 1000, 0.0001)
    s = spectrum(w, **kwargs)
    m0 = np.trapz(s, dx=0.0001)
    m2 = np.trapz((w ** 2) * s, dx=0.0001)
    m4 = np.trapz((w ** 4) * s, dx=0.0001)
    return np.sqrt(1 - (m2 ** 2) / (m0 * m4))


def formula(a, p):  # My formula for CDF of amplitudes
    term1 = (np.sqrt(2) * (1 + p ** 4)) / p * a  # Not sure about this term
    term2 = np.exp(-2 * a ** 2)
    term3 = (np.sqrt(2) * np.sqrt(1 - p ** 2)) / p * a
    return 0.5 * (1 - erf(term1) + term2 * (1 + erf(term3)))

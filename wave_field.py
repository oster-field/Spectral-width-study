"""Construction and storage of the distribution function of amplitudes/local extrema of a wave field with
different spectra."""
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functions import wave_field, zero_crossing, spectral_width
import os
import matplotlib.pyplot as plt
"""This is a block of parameters. Above are the parameters of the discrete inverse Fourier transform, below are the 
parameters of the spectra"""
# Param. of wave field
num_realizations = 400  # Number of realizations
max_w = 150  # The last frequency in spectrum
num_harmonics = 10 ** 4 + 2500  # Number of harmonics
# Param. of all spectra
spectrum_w0 = 2.5  # x-coord of peak
# My spectrum
power_spectrum = 6  # Power
# Gaussian spectrum
a = 1
b = 5
# Parabolic spectrum
kc = 1e-5
k = 1e5
# Extra parameters
ampl_or_extr = 'amplitude'  # CDF of amplitudes or extrema
name_of_spectrum = 'My spectrum'  # Name of spectrum
"""End of parameter block."""


def spectrum(w):  # My suggestion on max width spectrum
    if name_of_spectrum == 'My spectrum':
        return 1 / (1 + (w - spectrum_w0) ** power_spectrum)
    elif name_of_spectrum == 'Gaussian':
        return a * np.exp(-b * (w - spectrum_w0) ** 2)
    elif name_of_spectrum == 'Parabolic':
        q = ((k * kc) / (k + 2 * np.sqrt(k * kc) + kc)) ** (1 / 3)
        if spectrum_w0 - np.sqrt(q / kc) <= w <= spectrum_w0:
            return -kc * (w - spectrum_w0) ** 2 + q
        elif spectrum_w0 < w <= spectrum_w0 + np.sqrt(q / k):
            return -k * (w - spectrum_w0) ** 2 + q
        else:
            return 0


def run_wave_field(i):
    t, y, sx, sy, dw, dt = wave_field(max_w, num_harmonics, spectrum)
    if i == 0:  # Plot of obtained spectrum and ONE realization to check if parameters are correct
        plt.plot(sx, sy, color='red', linewidth=2, marker='.')
        plt.plot(sx, sx ** 4 * sy, color='green', linewidth=2, marker='.')
        plt.show()
        plt.plot(t, y, color='b', alpha=0.9, marker='.')
        plt.show()
    max_array = zero_crossing(t, y, ampl_or_extr)
    if i == 0:
        print(f'Number of individual waves: {len(max_array)}')
    significant_ampl = 2 * np.sqrt(np.trapz(sy, dx=dw))
    return max_array / significant_ampl


if __name__ == '__main__':
    CDF_points = np.array([])
    with ProcessPoolExecutor() as executor:  # Parallel computing of realizations
        results = list(tqdm(executor.map(run_wave_field, range(num_realizations)),
                            total=num_realizations, colour='green'))
    for result in results:
        CDF_points = np.append(CDF_points, result)
    epsilon = spectral_width(spectrum)
    CDF_y = np.linspace(1, 0, len(CDF_points), endpoint=False)
    CDF_x = np.sort(CDF_points)
    if not os.path.isdir(name_of_spectrum):
        os.mkdir(name_of_spectrum)
    save = {'x': CDF_x, 'y': CDF_y, 'width': epsilon}
    np.save(f'{name_of_spectrum}/R{num_realizations} MW{max_w} NH{num_harmonics}.npy', save)
    print(f'Saved, width = {epsilon}')

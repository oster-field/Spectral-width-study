"""Construction and storage of the distribution function of amplitudes/local extrema of a wave field with my
spectrum 1 / (1 + (x-x0)^p)."""
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functions import wave_field, zero_crossing, spectral_width
import os
import matplotlib.pyplot as plt

# Param. of wave field
num_realizations = 2000  # Number of realizations
max_w = 500  # The last frequency in spectrum
num_harmonics = 2 ** 14  # Number of harmonics
# Param. of spectra
spectrum_w0 = 0  # x-coord of peak
power_spectrum = 6  # Power
ampl_or_extr = 'amplitude'  # CDF of amplitudes or extrema
name_of_spectrum = 'My spectrum'  # Name of spectrum


def spectrum(w):  # My suggestion on max width spectrum
    return 1 / (1 + (w - spectrum_w0) ** power_spectrum)


def run_wave_field(i):
    t, y, sx, sy, dw, dt = wave_field(max_w, num_harmonics, spectrum)
    if i == 0:  # Plot of obtained spectrum and ONE realization to check if parameters are correct
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(sx, sy, color='red', linewidth=2)
        ax.plot(sx, sx ** 4 * sy, color='green', linewidth=2)
        plt.show()
        ax.plot(t, y, color='b', alpha=0.9)
        plt.show()
    max_array = zero_crossing(t, y, ampl_or_extr)
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
    np.save(f'{name_of_spectrum}/R{num_realizations} MW{max_w} NH{num_harmonics} MP{spectrum_w0} '
            f'P{power_spectrum}.npy', save)
    print(f'Saved, width = {epsilon}')

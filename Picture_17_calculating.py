from main import WaveFieldSimulation, f1, my_spectrum, parabolic_spectrum, optimize_spectrum, additional_function
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def objective(params, m, n):
    psi, alpha = params
    y_predict = additional_function(m, psi, alpha)
    return np.mean((n - y_predict) ** 2)


if __name__ == '__main__':
    if not os.path.isdir('Pic 17'):
        os.mkdir('Pic 17')
    spectrum_names = ['Parabolic', 'My spectrum']
    initial_guess = np.array([1, 1])
    result_x = np.arange(0)
    result_y_psi = np.arange(0)
    result_y_alpha = np.arange(0)
    """End of parameter block."""
    for spectrum_name in spectrum_names:
        if spectrum_name == 'Parabolic':
            num_harmonics = 8000
            widths = np.arange(0.15, 0.7, 0.05)
            max_w_array = -86.36 * widths + 72.95
        elif spectrum_name == 'My spectrum':
            num_harmonics = 10000
            widths = np.arange(0.7, 0.9, 0.05)
            max_w_array = -75 * widths + 92.5
        w0_opt, k_opt = 0, 0
        for i in tqdm(range(len(widths)), colour='red'):
            if spectrum_name == 'Parabolic':
                w0_opt, _, k_opt = optimize_spectrum(parabolic_spectrum, [0, 1e5, 1], [(0, None), (1e5, 1e5), (1e-6, None)], widths[i])
            elif spectrum_name == 'My spectrum':
                w0_opt, _ = optimize_spectrum(my_spectrum, [1, 6], [(0, None), (6, 6)], widths[i])
            simulation = WaveFieldSimulation(
                num_realizations=500,
                max_w=max_w_array[i],
                num_harmonics=num_harmonics,
                spectrum_w0=w0_opt,
                k=k_opt,
                name_of_spectrum=spectrum_name,
                showcase=False,
                save_file=False,
                progress=False
            )
            data = simulation.run_simulation()
            x = data.get('x')
            y = data.get('y')
            p = data.get('width')
            f = f1(x, p)
            y -= f
            result = minimize(objective, initial_guess, args=(x, y), bounds=[(0, None), (0, None)])
            psi_opt, alpha_opt = result.x
            result_x = np.append(result_x, p)
            result_y_psi = np.append(result_y_psi, psi_opt)
            result_y_alpha = np.append(result_y_alpha, alpha_opt)
            initial_guess = np.array([psi_opt, alpha_opt])
    np.save('Pic 17/p_values', result_x)
    np.save('Pic 17/psi_values', result_y_psi)
    np.save('Pic 17/alpha_values', result_y_alpha)
    # Preview
    plt.plot(result_x, result_y_psi)
    plt.show()

"""Construction of the distribution function of amplitudes/local extrema of a wave field with the selected spectrum
of the selected width. The program is used to control the quality of the wave field and fine-tune the
parameters of the discrete FFT"""
from main import WaveFieldSimulation, formula, my_spectrum, gaussian_spectrum, parabolic_spectrum, optimize_spectrum
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    """This is a block of parameters."""
    spectrum_name = 'Gaussian'
    width = 0.7  # Desired spectrum width
    # Here it is important to find a balance between these two parameters: so that the spectrum is optimally written
    # and the individual wave contains the required number of points
    num_harmonics = 30000  # Number of harmonics in summation
    max_w = 250  # Last frequency in the discrete Fourier spectrum

    num_realizations = 1  # Set 1 if CDF is not important
    """End of parameter block."""
    w0_opt, b_opt, k_opt = 0, 0, 0
    if spectrum_name == 'My spectrum':
        initial_guess = [1, 6]
        bounds = [(0, None), (6, 6)]
        w0_opt, _ = optimize_spectrum(my_spectrum, initial_guess, bounds, width)
    elif spectrum_name == 'Gaussian':
        initial_guess = [1, 1, 1]
        bounds = [(0, None), (1, 1), (1e-6, None)]
        w0_opt, _, b_opt = optimize_spectrum(gaussian_spectrum, initial_guess, bounds, width)
    elif spectrum_name == 'Parabolic':
        initial_guess = [0, 1e5, 1]
        bounds = [(0, None), (1e5, 1e5), (1e-6, None)]
        w0_opt, _, k_opt = optimize_spectrum(parabolic_spectrum, initial_guess, bounds, width)
    simulation = WaveFieldSimulation(
        num_realizations=num_realizations,
        max_w=max_w,
        num_harmonics=num_harmonics,
        spectrum_w0=w0_opt,
        b=b_opt,
        k=k_opt,
        ampl_or_extr='amplitude',
        name_of_spectrum=spectrum_name,
        showcase=True,
        save_file=False
    )
    data = simulation.run_simulation()
    x = data.get('x')
    y = data.get('y')
    p = data.get('width')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, np.exp(-2 * x ** 2), linewidth=2, color='black', label=f'Rayleigh')
    ax.plot(x, formula(x, p), linewidth=2, color='black', label=f'My formula', linestyle='dashed')
    ax.plot(x, y, linewidth=2, color='red', label=f'$F_{{a}}, ϵ = {np.round(p, 2)}$')
    ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
    ax.set_ylabel(f'CDF', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set(ylim=[0, 1])
    ax.set(xlim=[0, 2])
    ax.grid()
    ax.legend(fontsize=15)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.state('zoomed')
    plt.subplots_adjust(left=0.057, bottom=0.274, right=0.78, top=0.977, wspace=0.2, hspace=0.2)
    plt.show()

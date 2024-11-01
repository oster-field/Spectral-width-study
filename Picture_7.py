from main import WaveFieldSimulation, formula_higgins, my_spectrum, gaussian_spectrum, parabolic_spectrum, optimize_spectrum
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """This is a block of parameters."""
    spectrum_name = 'Parabolic'  # Gaussian, parabolic or my spectrum
    widths = np.array([0.2, 0.4, 0.7])  # Which widths we want to see in a picture
    num_realizations_array = np.array([2, 2, 2])  # This array is crucial for wave field and CDF quality
    max_w_array = np.array([200, 175, 150])  # This array is crucial for wave field and CDF quality
    num_harmonics_array = np.array([2 ** 13, 2 ** 13, 2 ** 13])  # This array is crucial for wave field and CDF quality
    """End of parameter block."""
    fig, ax = plt.subplots(1, len(widths))
    w0_opt, b_opt, k_opt = 0, 0, 0
    for i in range(len(widths)):
        if spectrum_name == 'My spectrum':
            initial_guess = [1, 6]
            bounds = [(0, None), (6, 6)]
            w0_opt, _ = optimize_spectrum(my_spectrum, initial_guess, bounds, widths[i])
        elif spectrum_name == 'Gaussian':
            initial_guess = [1, 1, 1]
            bounds = [(0, None), (1, 1), (1e-6, None)]
            w0_opt, _, b_opt = optimize_spectrum(gaussian_spectrum, initial_guess, bounds, widths[i])
        elif spectrum_name == 'Parabolic':
            initial_guess = [0, 1e5, 1]
            bounds = [(0, None), (1e5, 1e5), (1e-6, None)]
            w0_opt, _, k_opt = optimize_spectrum(parabolic_spectrum, initial_guess, bounds, widths[i])
        simulation = WaveFieldSimulation(
            num_realizations=num_realizations_array[i],
            max_w=max_w_array[i],
            num_harmonics=num_harmonics_array[i],
            spectrum_w0=w0_opt,
            b=b_opt,
            k=k_opt,
            name_of_spectrum=spectrum_name,
            ampl_or_extr='extrema',
            showcase=False,
            save_file=False
        )
        data = simulation.run_simulation()
        a = data.get('x')
        y = data.get('y')
        p = data.get('width')
        ax[i].plot(a, y, linewidth=3, color='r', label='$F$')
        ax[i].plot(a, np.exp(-2 * a ** 2), linewidth=2, color='black', label='$F_{R}$')
        ax[i].plot(a, formula_higgins(a, p), linewidth=2, color='black', linestyle='dashed',
                   label='$F_{M}$')
        ax[i].tick_params(labelsize=20)
        ax[i].set(ylim=[0, 1])
        ax[i].set(xlim=[0, 1.75])
        ax[i].grid()
        ax[i].legend(title=f'ϵ = {widths[i]}', fontsize=15, title_fontproperties={'size': 16, 'weight': 'semibold'})
    ax[1].set_xlabel('Normalized value of local maxima', fontsize=20)
    ax[0].set_ylabel(f'CDF', fontsize=20)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.state('zoomed')
    plt.subplots_adjust(left=0.057, bottom=0.274, right=0.78, top=0.977, wspace=0.2, hspace=0.2)
    plt.show()


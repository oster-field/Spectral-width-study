from main import WaveFieldSimulation, f1, my_spectrum, gaussian_spectrum, parabolic_spectrum, optimize_spectrum
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    """This is a block of parameters."""
    spectrum_name = 'My spectrum'  # Gaussian, parabolic or my spectrum
    widths = np.array([0.2, 0.4, 0.7])  # Which widths we want to see in a picture
    num_realizations_array = np.array([1000, 1000, 1000])  # This array is crucial for CDF quality
    max_w_array = np.array([105, 125, 200, 250])  # Optimal values for: (obtained from CDF creation.npy)
    # Gaussian: 100 | 60 | 40
    # Parabolic: 50 | 25 | 12.5
    # My spectrum: 200 | 90 | 40
    num_harmonics_array = np.array([3000, 5000, 16000])  # Optimal values for:
    # Parabolic: 8000
    # Gaussian: 9000
    # My spectrum: 10000
    """End of parameter block."""
    colors = ["red", "green", "blue", "black"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if not os.path.isdir('Pic 15'):
        os.mkdir('Pic 15')
    w0_opt, b_opt, k_opt = 0, 0, 0
    for i, width in enumerate(widths):
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
            num_realizations=num_realizations_array[i],
            max_w=max_w_array[i],
            num_harmonics=num_harmonics_array[i],
            spectrum_w0=w0_opt,
            b=b_opt,
            k=k_opt,
            name_of_spectrum=spectrum_name,
            showcase=False,
            save_file=False
        )
        data = simulation.run_simulation()
        a = data.get('x')
        y = data.get('y')
        p = data.get('width')
        ax.plot(a, y, linewidth=3, color=colors[i], label=f'$F_{{a}}, ϵ = {widths[i]}$')
        f = f1(a, p)
        ax.plot(a, f, linestyle='dotted', linewidth=2, color=colors[i], label=f'$F_{{1}}, ϵ = {widths[i]}$')
        ax.plot(a, y - f, linestyle='dashed', linewidth=2, color=colors[i], label=f'$F_{{a}}-F_{{1}}, ϵ = {widths[i]}$')
        save = {'x': a, 'y': y - f, 'width': p}
        np.save(f'Pic 15/F_A-F1 eps{widths[i]}.npy', save)
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

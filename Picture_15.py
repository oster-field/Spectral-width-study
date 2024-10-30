from main import WaveFieldSimulation, f1
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    """This is a block of parameters."""
    widths = np.array([0.2, 0.4, 0.7])  # What widths we want to see in a picture
    num_realizations = 1000
    max_w = 600
    num_harmonics = 2 ** 13
    """End of parameter block."""
    initial_guess = [1, 0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if not os.path.isdir('Pic 15'):
        os.mkdir('Pic 15')
    for i in range(len(widths)):
        def target(params):
            b, w0 = params
            m0_val, _ = quad(lambda x: np.exp(-b * (x - w0) ** 2), 0, max_w)
            m2_val, _ = quad(lambda x: x ** 2 * np.exp(-b * (x - w0) ** 2), 0, max_w)
            m4_val, _ = quad(lambda x: x ** 4 * np.exp(-b * (x - w0) ** 2), 0, max_w)
            if m0_val == 0 or m4_val == 0:
                return np.inf
            ratio = np.sqrt(1 - m2_val ** 2 / (m0_val * m4_val))
            return (ratio - widths[i]) ** 2


        result = minimize(target, initial_guess, bounds=[(0, None), (None, None)])
        b_opt, w0_opt = result.x
        simulation = WaveFieldSimulation(
            num_realizations=num_realizations,
            max_w=max_w,
            num_harmonics=num_harmonics,
            spectrum_w0=w0_opt,
            b=b_opt,
            name_of_spectrum='Gaussian',
            showcase=False,
            save_file=False
        )
        data = simulation.run_simulation()
        a = data.get('x')
        y = data.get('y')
        p = data.get('width')
        colors = ["red", "green", "blue"]
        ax.plot(a, y, linewidth=2, marker='.', color=colors[i], label=f'$F_{{a}}, ϵ = {widths[i]}$')
        f = f1(a, p)
        ax.plot(a, f, linestyle='dotted', linewidth=2, color=colors[i], label=f'$F_{{1}}, ϵ = {widths[i]}$')
        ax.plot(a, y - f, linestyle='dashed', linewidth=2, color=colors[i], label=f'$F_{{a}}-F_{{1}}, ϵ = {widths[i]}$')
        save = {'x': a, 'y': y - f, 'width': p}
        np.save(f'Pic 15/F_A-F1 eps{widths[i]}.npy', save)
        max_w += 50
        num_harmonics += 10000
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

from main import additional_function, WaveFieldSimulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def objective(psi, m, n):
    y_predict = additional_function(m, psi[0])
    return np.mean((n - y_predict) ** 2)


if __name__ == '__main__':
    initial_psi = [1]
    w0 = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(10):
        simulation = WaveFieldSimulation(
            num_realizations=20,
            max_w=200,
            num_harmonics=2 ** 13,
            spectrum_w0=w0,
            a=1,
            b=5,
            ampl_or_extr='amplitude',
            name_of_spectrum='Gaussian',
            showcase=False,
            save_file=False
        )
        data = simulation.run_simulation()
        x = data.get('x')
        y = data.get('y')
        p = data.get('width')
        result = minimize(objective, initial_psi, args=(x, y), bounds=[(0, None)])
        psi_opt = result.x[0]
        ax.scatter(p, psi_opt, color='g', s=2.5)
        initial_psi = [psi_opt]
        w0 += 0.5
    ax.scatter([], [], linewidth=2, color='g', label='$ψ$')
    ax.set_xlabel('ϵ', fontsize=20)
    ax.set_ylabel('ψ', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid()
    ax.legend(fontsize=15)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.state('zoomed')
    plt.subplots_adjust(left=0.057, bottom=0.274, right=0.78, top=0.977, wspace=0.2, hspace=0.2)
    plt.show()

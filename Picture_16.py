import matplotlib.pyplot as plt
import os
import numpy as np
from main import additional_function
from scipy.optimize import minimize


def objective(psi, m, n):
    y_predict = additional_function(m, psi[0])
    return np.mean((n - y_predict) ** 2)


fig = plt.figure()
ax = fig.add_subplot(111)
colors = ["red", "green", "blue"]
i = 0
for i, filename in enumerate(os.listdir('Pic 15')):
    file_path = os.path.join('Pic 15', filename)
    loaded_data = np.load(file_path, allow_pickle=True)
    x = loaded_data.item().get('x')
    y = loaded_data.item().get('y')
    p = loaded_data.item().get('width')
    ax.plot(x, y, linestyle='dashed', linewidth=2, color=colors[i], label=f'$F_{{a}}-F_{{1}}, ϵ = {np.round(p, 2)}$')
    initial_psi = [1]
    result = minimize(objective, initial_psi, args=(x, y), bounds=[(0, None)])
    psi_opt = result.x[0]
    ax.plot(x, additional_function(x, psi_opt), linewidth=4, color=colors[i],
            alpha=0.5, label=f'$γ, ψ = {np.round(psi_opt, 2)}$')
ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
ax.set_ylabel(f'CDF', fontsize=20)
ax.tick_params(labelsize=20)
ax.set(ylim=[0, 0.5])
ax.set(xlim=[0, 1])
ax.grid()
ax.legend(fontsize=15)
fig_manager = plt.get_current_fig_manager()
fig_manager.window.state('zoomed')
plt.subplots_adjust(left=0.057, bottom=0.274, right=0.78, top=0.977, wspace=0.2, hspace=0.2)
plt.show()

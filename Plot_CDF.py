"""Plotting previously saved graphs for the CDF"""
import matplotlib.pyplot as plt
import numpy as np
from functions import formula

filenames = ['My spectrum/R400 MW150 NH12500']  # Input filenames
colors = ["red", "green", "blue", "#F7CAC9", "#92A8D1",
          "#955251", "#B565A7", "#009B77", "#DD4124", "#45B8AC"]
labels = ['Gaussian', 'My']
fig = plt.figure()
ax = fig.add_subplot(111)


for i in range(len(filenames)):
    loaded_data = np.load(f'{filenames[i]}.npy', allow_pickle=True)
    x = loaded_data.item().get('x')
    y = loaded_data.item().get('y')
    p = loaded_data.item().get('width')
    ax.scatter(x, y, s=2.5, color=colors[i])
    ax.plot([], [], linewidth=6, color=colors[i], label=labels[i])  # Better legend appearance
ax.plot(x, np.exp(-2 * x ** 2), linewidth=2, color='black', label='Rayleigh distribution')
ax.plot(x, formula(x, p), linewidth=2, color='black', linestyle='dashed', label='My distribution')
ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
ax.set_ylabel(f'CDF, p={np.round(p, 2)}', fontsize=20)
ax.tick_params(labelsize=20)
ax.set(ylim=[0, 1])
ax.set(xlim=[0, 2])
ax.legend(fontsize=15, title=f'ν={np.round(0, 2)}, ϵ={np.round(p, 2)}, ρ={np.round(0, 2)}', title_fontsize='15')
ax.grid()
fig_manager = plt.get_current_fig_manager()
fig_manager.window.state('zoomed')
plt.subplots_adjust(left=0.057, bottom=0.274, right=0.78, top=0.977, wspace=0.2, hspace=0.2)
plt.show()

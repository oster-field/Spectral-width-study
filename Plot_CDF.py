"""Plotting previously saved graphs for the CDF"""
import matplotlib.pyplot as plt
import numpy as np
from functions import formula

spectrum_name = 'My spectrum'  # Input spectrum name
filename = 'R2000 MW400 NH8192 MP10 P4'  # Input filename

loaded_data = np.load(f'{spectrum_name}/{filename}.npy', allow_pickle=True)
x = loaded_data.item().get('x')
y = loaded_data.item().get('y')
p = loaded_data.item().get('width')
fig = plt.figure()
ax = fig.add_subplot(111)
# Plotting
ax.plot(x, y, linewidth=2.5, color='red', label='none')
ax.plot(x, np.exp(-2 * x ** 2), linewidth=2, color='black', label='Rayleigh distribution')
ax.plot(x, formula(x, p), linewidth=2, color='black', linestyle='dashed', label='Rayleigh distribution')

ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
ax.set_ylabel(f'CDF, p={np.round(p, 2)}', fontsize=20)
ax.tick_params(labelsize=20)
ax.set(ylim=[0, 1])
ax.set(xlim=[0, 2])
ax.legend(fontsize=15, title=f'ν={np.round(0,2)}, ϵ={np.round(p,2)}, ρ={np.round(0,2)}', title_fontsize='15')
ax.grid()
fig_manager = plt.get_current_fig_manager()
fig_manager.window.state('zoomed')
plt.subplots_adjust(left=0.057, bottom=0.274, right=0.78, top=0.977, wspace=0.2, hspace=0.2)
plt.show()

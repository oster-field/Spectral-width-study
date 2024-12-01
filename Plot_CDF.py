"""Plotting previously saved graphs for the CDF"""
import matplotlib.pyplot as plt
import numpy as np
from main import formula
from scipy.special import erf
"""This is a block of parameters."""
filenames = ['My spectrum/R1000 MW70 NH10000']  # Input filenames in format SpectrumName/FileName
labels = ['Gaussian', 'My']  # And their labels
"""End of parameter block."""

colors = ["red", "green", "blue", "#F7CAC9", "#92A8D1",
          "#955251", "#B565A7", "#009B77", "#DD4124", "#45B8AC"]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
for i in range(len(filenames)):
    loaded_data = np.load(f'{filenames[i]}.npy', allow_pickle=True)
    x = loaded_data.item().get('x')
    y = loaded_data.item().get('y')
    p = loaded_data.item().get('width')
    ax.plot(x, y, linewidth=2.5, color=colors[i])
    ax.plot(x, 1 - np.tanh(1.2 * x))
ax.tick_params(labelsize=20)
fig_manager = plt.get_current_fig_manager()
fig_manager.window.state('zoomed')
plt.subplots_adjust(left=0.057, bottom=0.274, right=0.78, top=0.977, wspace=0.2, hspace=0.2)
plt.show()

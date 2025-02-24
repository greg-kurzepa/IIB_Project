import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

s = _pile_and_soil.Soil()
p = _pile_and_soil.Pile(0.15, 10, 2275, E=20e9) # https://ukrstarline.ua/en/reinforced-concrete-products/reinforced-concrete-pile-driven/reinforced-concrete-pile-driven-c-10030-10
system = _pile_and_soil.System(p, s)

N = 101 # number of nodes along pile
pile_length = system.p.L

fig, ax = plt.subplots()
z = np.linspace(0, pile_length, N)
line, = ax.plot(system.soil_limit_model(z), z, lw=2)
ax.set_ylim(0, pile_length)
ax.invert_yaxis()
plt.grid()

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the pile length.
ax_len = fig.add_axes([0.25, 0.2, 0.65, 0.03])
len_slider = Slider(
    ax=ax_len,
    label='Pile Length L',
    valmin=1,
    valmax=100,
    valinit=pile_length,
)

def update(val):
    z = np.linspace(0, len_slider.val, N)
    system.p.L = len_slider.val
    F = system.soil_limit_model(z)
    line.set_ydata(z)
    line.set_xdata(F)
    ax.set_ylim(0, len_slider.val)
    ax.set_xlim(F.min(), F.max())
    ax.invert_yaxis()
    fig.canvas.draw_idle()

len_slider.on_changed(update)

plt.show()
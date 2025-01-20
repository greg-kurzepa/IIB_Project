import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import _pile_and_soil
import _model_shooting
import _model_springs

s = _pile_and_soil.Soil()
p = _pile_and_soil.Pile(0.15, 10, 2275, E=20e9) # https://ukrstarline.ua/en/reinforced-concrete-products/reinforced-concrete-pile-driven/reinforced-concrete-pile-driven-c-10030-10
system = _pile_and_soil.System(p, s)

P = 200e3 # top axial load
N = 101 # number of nodes along pile

def slider_wrapper(pile_length, alpha, model):
    system.p.L = pile_length
    system.s.alpha = alpha

    if model == "shooting":
        return _model_shooting.solve(system, P, N)
    elif model == "springs":
        return _model_springs.solve(system, P, N)
    elif model == "springs_stable":
        return _model_springs.solve_stable(system, P, N)
    else:
        raise ValueError("model must be 'shooting' or 'springs'")

pile_length = system.p.L
alpha = system.s.alpha

fig, ax = plt.subplots(1,2, sharey=True)
z1, F1, strain1, u1 = slider_wrapper(pile_length, alpha, model="springs")
z2, F2, strain2, u2 = slider_wrapper(pile_length, alpha, model="springs_stable")
line11, = ax[0].plot(F1, z1, lw=2, label="Springs")
line12, = ax[0].plot(F2, z2, lw=2, label="Springs Stable", linestyle="--")
line21, = ax[1].plot(u1, z1, lw=2, label="Springs")
line22, = ax[1].plot(u2, z2, lw=2, label="Springs Stable", linestyle="--")
ax[0].set_ylabel('Depth z')
ax[0].invert_yaxis()
ax[0].set_xlabel('Force F')
ax[1].set_xlabel('Displacement u')
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.35)

# Make a horizontal slider to control the pile length.
ax_len = fig.add_axes([0.25, 0.2, 0.65, 0.03])
len_slider = Slider(
    ax=ax_len,
    label='Pile Length L',
    valmin=1,
    valmax=100,
    valinit=pile_length,
)

# Make a horizontal slider to control alpha.
ax_alpha = fig.add_axes([0.25, 0.1, 0.65, 0.03])
alpha_slider = Slider(
    ax=ax_alpha,
    label='Alpha',
    valmin=0,
    valmax=1,
    valinit=alpha,
)

# The function to be called anytime a slider's value changes
def update(val):
    z1, F1, _, u1 = slider_wrapper(len_slider.val, alpha_slider.val, model="springs")
    z2, F2, _, u2 = slider_wrapper(len_slider.val, alpha_slider.val, model="springs_stable")
    line11.set_xdata(F1)
    line11.set_ydata(z1)
    line12.set_xdata(F2)
    line12.set_ydata(z2)
    line21.set_xdata(u1)
    line21.set_ydata(z1)
    line22.set_xdata(u2)
    line22.set_ydata(z2)
    fig.canvas.draw_idle()
    ax[0].set_xlim(min(F1[:-1].min(), F2[:-1].min()), max(F1[:-1].max(), F2[:-1].max()))
    ax[0].set_ylim(0, len_slider.val)
    ax[1].set_xlim(min(u1[:-1].min(), u2[:-1].min()), max(u1[:-1].max(), u2[:-1].max()))
    ax[0].invert_yaxis()

# register the update function with each slider
len_slider.on_changed(update)
alpha_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    len_slider.reset()
    alpha_slider.reset()
button.on_clicked(reset)

plt.show()
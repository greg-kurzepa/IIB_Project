#%%

import numpy as np
import scipy
import scipy.integrate

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

#%%

m_to_in = 39.3701

class Pile:
    def __init__(self, R, L, W, E):
        self.R = R # the outer radius of the pile
        self.L = L # the length of the pile
        self.W = W # the total weight of the pile
        self.A = np.pi*self.R**2 # FOR NOW non-hollow
        self.C = 2*np.pi*self.R
        self.E = E

class Soil:
    def __init__(self, alpha=0.4, gamma=20e3, N_c=9, s_u0=30e3, rho=4.8e3, sigma_n=5e3):
        self.alpha = alpha
        self.gamma = gamma
        self.N_c = N_c
        self.s_u0 = s_u0
        self.rho = rho
        self.sigma_n = sigma_n

    def soil_limit_model(self, pile, z):
        # base bearing load
        BEARING = np.pi * pile.R**2 * ( self.gamma*pile.L + self.N_c*( self.s_u0 + self.rho*pile.L ) )

        # total vertical shear force below z
        SHEAR = 2*np.pi*pile.R*self.alpha * ( 0.5*self.rho*( pile.L**2 - z**2 ) + self.s_u0*( pile.L - z ) )

        # total pile weight below z
        W_FRAC = pile.W * (1 - z/pile.L)

        # vertical force in steel at that point
        F = BEARING + SHEAR - W_FRAC
        return F
    
    def g(self, disp_over_D):
        # values to interpolate from
        z_over_D = np.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, np.inf]) / m_to_in
        tau_over_tau_ult = np.array([0, 0.3, 0.5, 0.75, 0.9, 1, 0.9, 0.9])

        # return np.interp(disp_over_D, z_over_D, tau_over_tau_ult)
        return np.sign(disp_over_D)*np.interp(np.abs(disp_over_D), z_over_D, tau_over_tau_ult)

    def h(self, disp_over_D):
        # values to interpolate from
        z_over_D = np.array([-np.inf, 0, 0.002, 0.013, 0.042, 0.073, 0.1, np.inf]) / m_to_in
        Q_over_Q_ult = np.array([0, 0, 0.25, 0.5, 0.75, 0.9, 1, 1])

        # return np.interp(disp_over_D, z_over_D, Q_over_Q_ult)
        return disp_over_D*np.interp(disp_over_D, z_over_D, Q_over_Q_ult)

#%%

def solve(pile, soil, P):
    z = np.linspace(0, pile.L, N)
    dz = pile.L / (N-1) # length of one element

    # Create initial guess using soil limit model
    soil_failure_F = soil.soil_limit_model(pile, z)
    soil_failure_F *= (P / soil_failure_F[0])
    strain_guess = - soil_failure_F / (pile.A*pile.E)
    strain_top = -P/(pile.A*pile.E)
    disp_guess = np.cumsum(strain_guess * dz)
    disp_guess -= disp_guess[-1]
    print("dispguess: ", disp_guess[-1])
    y_guess = np.stack([disp_guess, strain_guess])

    tau_ult = lambda z: soil.alpha * (soil.s_u0 + soil.rho*z) # ultimate skin friction
    Q_ult = soil.N_c * (soil.s_u0 + soil.rho * pile.L) # ultimate end bearing pressure

    # runge-kutta function. dy/dz = f(z,y)
    # y = (u, du/dz)
    def diffeq(z, y):
        ret1 = y[1]
        ret2 = np.append(
                # note, will want to make alpha calculation better as per guide
                # all but the last node are soil side shear. the last node is soil bearing shear.
                pile.C * tau_ult(z)[:-1] * soil.g(y[0][:-1] / (2*pile.R)) / (pile.A*pile.E),
                Q_ult * soil.h(y[0][-1] / (2*pile.R)) / (pile.A*pile.E*dz)
            )
        # print(z.shape, y.shape, ret2.shape)
        
        return np.vstack((ret1, ret2))

    # residual definition. i.e. returns zero when solution matches boundary conditions.
    # ya = [u, du/dz] at top. yb = [u, du/dz] at bottom.
    def bc(ya, yb):
        return np.array([
            Q_ult * soil.h(yb[0] / (2*pile.R)) - (-pile.A*pile.E*yb[1]), # satisfy F = h(u) at bottom, i.e. h(u) - F = 0
            ya[1] - strain_top # the value of strain, du/dz, at z=0 will be set to strain_top
        ])

    # Solve boundary value problem
    result = scipy.integrate.solve_bvp(diffeq, bc, z, y_guess, tol=1e-10)

    u = result.sol(z)[0]
    strain = result.sol(z)[1]
    F = result.sol(z)[1] * pile.A*pile.E

    return z, F, u

soil = Soil()
pile = Pile(0.15, 10, 2275, E=20e9) # https://ukrstarline.ua/en/reinforced-concrete-products/reinforced-concrete-pile-driven/reinforced-concrete-pile-driven-c-10030-10

P = 200e3 # top axial load
N = 101 # number of nodes along pile

# F = solve(pile, soil, P, z)

# # guessed strain boundary condition at z=L for shooting method
# # using the soil-limit failure model. we know the shape is quadratic so need evaluate only 3 points.
# F_bottom = soil.soil_limit_model(pile, pile.L) # evaluate force at z=L
# F_bottom_scaled = F_bottom * P/soil.soil_limit_model(pile, 0) # evaluate force at z=0 and scale F_bottom by the ratio of this to actual top load
# strain_guess_bottom = - F_bottom_scaled / (pile.A*pile.E)

#%%

# Plot!

# plt.plot(F, z)
# plt.xlabel("sol")
# plt.ylabel("z")
# plt.grid()
# plt.gca().invert_yaxis()
# plt.show()
# %%

def slider_wrapper(pile, pile_length, alpha):
    pile.L = pile_length
    soil.alpha = alpha

    return solve(pile, soil, P)

pile_length = pile.L
alpha = soil.alpha

fig, ax = plt.subplots(1,2, sharey=True)
z, F, u = slider_wrapper(pile, pile_length, alpha)
line1, = ax[0].plot(F, z, lw=2)
line2, = ax[1].plot(u, z, lw=2)
ax[0].set_ylabel('Depth z')
ax[0].invert_yaxis()
ax[0].set_xlabel('Force F')
ax[1].set_xlabel('Displacement u')
ax[0].grid()
ax[1].grid()

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.35)

# Make a horizontal slider to control the frequency.
ax_len = fig.add_axes([0.25, 0.2, 0.65, 0.03])
len_slider = Slider(
    ax=ax_len,
    label='Pile Length L',
    valmin=1,
    valmax=100,
    valinit=pile_length,
)

# Make a horizontal slider to control the frequency.
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
    _, F, u = slider_wrapper(pile, len_slider.val, alpha_slider.val)
    line1.set_xdata(F)
    line2.set_xdata(u)
    fig.canvas.draw_idle()
    ax[0].set_xlim(F[:-1].min(), F[:-1].max())
    ax[1].set_xlim(u[:-1].min(), u[:-1].max())

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
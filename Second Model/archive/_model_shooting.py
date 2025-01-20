import numpy as np
import scipy
import scipy.integrate

def solve(system, P, N=100, tol=1e-8):
    z = np.linspace(0, system.p.L, N)
    dz = system.p.L / (N-1) # length of one element

    # Create initial guess using soil limit model
    soil_failure_F = system.soil_limit_model(z)
    soil_failure_F *= (P / soil_failure_F[0])
    strain_guess = - soil_failure_F / (system.p.A*system.p.E)
    strain_top = -P/(system.p.A*system.p.E)
    disp_guess = np.cumsum(strain_guess * dz)
    disp_guess -= disp_guess[-1]
    y_guess = np.stack([disp_guess, strain_guess])

    tau_ult = lambda z: system.s.alpha * (system.s.s_u0 + system.s.rho*z) # ultimate skin friction
    Q_ult = system.s.N_c * (system.s.s_u0 + system.s.rho * system.p.L) # ultimate end bearing pressure

    # runge-kutta function. dy/dz = f(z,y)
    # y = (u, du/dz)
    def diffeq(z, y):
        ret1 = y[1]
        ret2 = np.append(
                # note, will want to make alpha calculation better as per guide
                # all but the last node are soil side shear. the last node is soil bearing shear.
                system.p.C * tau_ult(z)[:-1] * system.g(y[0][:-1]) / (system.p.A*system.p.E),
                Q_ult * system.h(y[0][-1]) / (system.p.A*system.p.E*dz)
            )
        # print(z.shape, y.shape, ret2.shape)
        
        return np.vstack((ret1, ret2))

    # residual definition. i.e. returns zero when solution matches boundary conditions.
    # ya = [u, du/dz] at top. yb = [u, du/dz] at bottom.
    def bc(ya, yb):
        return np.array([
            Q_ult * system.h(yb[0]) - (-system.p.A*system.p.E*yb[1]), # satisfy F = h(u) at bottom, i.e. h(u) - F = 0
            ya[1] - strain_top # the value of strain, du/dz, at z=0 will be set to strain_top
        ])

    # Solve boundary value problem
    result = scipy.integrate.solve_bvp(diffeq, bc, z, y_guess, tol=tol)

    u = result.sol(z)[0]
    strain = result.sol(z)[1]
    F = - result.sol(z)[1] * system.p.A*system.p.E

    return z, F, strain, u
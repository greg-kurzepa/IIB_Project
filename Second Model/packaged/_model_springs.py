import numpy as np
import scipy
import scipy.optimize

def solve_stable(system, P, N=100, tol=1e-8):
    z = np.linspace(0, system.p.L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:])
    dz = system.p.L / (N-1) # length of one element

    # Define intermediate constants
    # Q_ult = system.s.N_c * (system.s.s_u0 + system.s.rho * system.p.L) # WRONG ultimate end bearing pressure
    Q_ult = np.pi * system.p.A * (system.s.gamma * system.p.L + system.s.N_c * (system.s.s_u0 + system.s.rho * system.p.L)) # ultimate end bearing pressure
    k = 2 * system.p.A * system.p.E / dz

    # Non-dimensionalise initial guess
    v_guess = np.zeros_like(z_midpoints)

    # Define non-dimensional constants
    a1 = system.s.alpha * system.s.s_u0 * system.p.C * dz / P
    a2 = system.s.alpha * system.s.rho * system.p.C * dz / P
    a3 = Q_ult / (2*P)
    a4 = 2 * P / k # not / (k*D) since h,g already account for D
    b = a1 + a2 * z_midpoints

    def f_simultaneous(v):
        # get LHS of simultaneous equations
        lhs = 1 - np.cumsum(b * system.g(a4*v))

        # get RHS of simultaneous equations
        c1 = v[:-1] - v[1:]
        c2 = scipy.optimize.fsolve(
            # lambda disp_N : k * (mu[-1] - disp_N) - Q_ult * system.h(disp_N), # solve this
            lambda w : v[-1] - w - a3 * system.h(a4*w),
            v[-1] # starting guess
        )
        rhs = np.append(c1, 2*(v[-1] - c2))

        # returns vector size N-1 = size(mu)
        return lhs - rhs
    
    v = scipy.optimize.fsolve(f_simultaneous, v_guess, xtol=tol)
    mu = (2 * P / k) * v
    tau_ult = lambda z: system.s.alpha * (system.s.s_u0 + system.s.rho*z) # ultimate skin friction
    shear = system.p.C * dz * tau_ult(z_midpoints) * system.g(mu)
    a = np.cumsum(np.insert(shear, 0, 0))
    F = P - a
    strain = F / (system.p.A * system.p.E)
    u = np.insert(mu - F[1:] / k, 0, mu[0] + P / k)
    # TODO: as a sanity check, implement comparison of u from e1 and e2 :)

    return z, F, strain, u
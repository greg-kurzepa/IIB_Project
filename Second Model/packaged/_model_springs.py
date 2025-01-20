import numpy as np
import scipy
import scipy.optimize

def solve_stable(system, P, N=100, tol=1e-8):
    z = np.linspace(0, system.p.L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:])
    dz = system.p.L / (N-1) # length of one element

    # Define intermediate constants
    Q_ult = system.s.N_c * (system.s.s_u0 + system.s.rho * system.p.L) # ultimate end bearing pressure
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

# def solve(system, P, N=100, tol=1e-8):
#     z = np.linspace(0, system.p.L, N)
#     z_midpoints = 0.5 * (z[:-1] + z[1:])
#     dz = system.p.L / (N-1) # length of one element

#     # Create initial guess using soil limit model [ERROR PRONE]
#     # soil_failure_F = system.soil_limit_model(z)
#     # soil_failure_F *= (P / soil_failure_F[0])
#     # strain_guess = - soil_failure_F / (system.p.A*system.p.E)
#     # strain_top = -P/(system.p.A*system.p.E)
#     # disp_guess = np.cumsum(strain_guess * dz)
#     # disp_guess -= disp_guess[-1]
#     # mu_guess = disp_guess[:-1]
    
#     # Initial guess of zero
#     mu_guess = np.zeros_like(z_midpoints)

#     tau_ult = lambda z: system.s.alpha * (system.s.s_u0 + system.s.rho*z) # ultimate skin friction
#     Q_ult = system.s.N_c * (system.s.s_u0 + system.s.rho * system.p.L) # ultimate end bearing pressure
#     k = 2 * system.p.A * system.p.E / dz
#     # print(f"k: {k}")

#     def f_simultaneous(mu):
#         # get LHS of simultaneous equations
#         shear = system.p.C * dz * tau_ult(z_midpoints) * system.g(mu)
#         force_lhs = P - np.cumsum(shear)

#         # get RHS of simultaneous equations
#         b = 0.5 * k * (mu[:-1] - mu[1:])
#         c = scipy.optimize.fsolve(
#             lambda disp_N : k * (mu[-1] - disp_N) - Q_ult * system.h(disp_N), # solve this
#             mu[-1] # starting guess
#         )
#         force_rhs = np.append(b, Q_ult * system.h(c))

#         # returns vector size N-1 = size(mu)
#         return force_lhs - force_rhs
    
#     mu = scipy.optimize.fsolve(f_simultaneous, mu_guess, xtol=tol)
#     shear = system.p.C * dz * tau_ult(z_midpoints) * system.g(mu)
#     a = np.cumsum(np.insert(shear, 0, 0))
#     F = P - a
#     strain = F / (system.p.A * system.p.E)
#     u = np.insert(mu - F[1:] / k, 0, mu[0] + P / k)
#     # TODO: as a sanity check, implement comparison of u from e1 and e2 :)

#     return z, F, strain, u
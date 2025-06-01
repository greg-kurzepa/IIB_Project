#%%

import numpy as np
import scipy
import scipy.integrate
import scipy.optimize

import packaged.utility as utility

import logging
logger = logging.getLogger(__name__)

m_to_in = 39.3701
gamma_w = 9.81e3 # unit weight of water
# Es = 200e9 # A *reference* young's modulus value for steel, used for non-dimensionalising (for numerical accuracy)

def tau_over_tau_ult_clay(disp_over_D, t_res=0.9):
    # values to interpolate from
    z_over_D = np.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, np.inf]) / m_to_in
    tau_over_tau_ult = np.array([0, 0.3, 0.5, 0.75, 0.9, 1, t_res, t_res])

    # below works for positive AND negative displacement.
    return np.sign(disp_over_D)*np.interp(np.abs(disp_over_D), z_over_D, tau_over_tau_ult)

def tau_over_tau_ult_sand(disp_over_D):
    # values to interpolate from
    z_over_D = np.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, np.inf]) / m_to_in
    tau_over_tau_ult = np.array([0, 0.3, 0.5, 0.75, 0.9, 1, 1, 1])

    # below works for positive AND negative displacement.
    return np.sign(disp_over_D)*np.interp(np.abs(disp_over_D), z_over_D, tau_over_tau_ult)

def Q_over_Q_ult(disp_over_D):
    # values to interpolate from
    z_over_D = np.array([-np.inf, 0, 0.002, 0.013, 0.042, 0.073, 0.1, np.inf]) / m_to_in
    Q_over_Q_ult = np.array([0, 0, 0.25, 0.5, 0.75, 0.9, 1, 1])

    # below works for only downwards (+ve) displacement since going up there is no resistance.
    return np.interp(disp_over_D, z_over_D, Q_over_Q_ult)

def second_order_central_difference(u, dz):
    return ( -2*u[1:-1] + u[:-2] + u[2:] ) / dz**2

def f_acceleration(z, u, zs, pile_A, pile_C, pile_E, pile_D, l_layer_type, l_base_depth, t_res_clay, shaft_pressure_limit, tau_ult_over_AEm):
    """Finds vector d2u/dz2 for a vector input of depth z, also supports single values."""
    
    # print(z)

    layer_ids = np.searchsorted(l_base_depth, z)
    tau_ult_over_AEm = np.interp(z, zs, tau_ult_over_AEm)
    shaft_pressure_limit = np.interp(z, zs, shaft_pressure_limit)
    A = np.interp(z, zs, pile_A)
    C = np.interp(z, zs, pile_C)
    D = np.interp(z, zs, pile_D)
    try:
        tau_over_AEm = tau_ult_over_AEm * np.where(
            l_layer_type[layer_ids] == 0, # 0 for clay, 1 for sand
            tau_over_tau_ult_clay(u / D, t_res=t_res_clay),
            tau_over_tau_ult_sand(u / D)
        )
    except IndexError as e:
        print(z, l_base_depth, layer_ids)
        raise IndexError from e
    tau_over_AEm = np.clip(tau_over_AEm, -shaft_pressure_limit/(A*pile_E), shaft_pressure_limit/(A*pile_E))

    return C * tau_over_AEm

class SolveData():
    def __init__(self, F, strain, u, zero_equilibrium, l2loss, tau, Q,
                 shaft_pressure_limit, Q_limit, eff_stress, tau_ult, Q_ult, P, P_cap,
                 Q_cap=None, S_cap=None, too_light=False):
        
        self.F = F
        self.strain = strain
        self.u = u
        self.zero_equilibrium = zero_equilibrium
        self.l2loss = l2loss
        self.tau = tau
        self.Q = Q

        self.shaft_pressure_limit = shaft_pressure_limit
        self.Q_limit = Q_limit
        self.eff_stress = eff_stress
        self.tau_ult = tau_ult
        self.Q_ult = Q_ult
        self.P = P
        self.P_cap = P_cap

        self.Q_cap = Q_cap
        self.S_cap = S_cap

        self.too_light = too_light

def forward_api(pile_D, pile_L, pile_E, # pile parameters
                l_layer_type, l_gamma_d, l_e, l_c1, l_c2, l_shaft_pressure_limit, l_end_pressure_limit, l_base_depth, # soil parameters.
                P, z_w, N=100, t_res_clay=0.9, tol=1e-8, do_print=False, custom_args=None, max_nodes=500):
    """Implementation of RSPile axially loaded 1D FEM pile/soil stress/strain model using spring discretisation.
    tau_ult and q_ult calculations for sands and clays are taken from API2GEO 2011 (Reading 8.2).
    https://static.rocscience.cloud/assets/verification-and-theory/RSPile/RSPile-Axially-Loaded-Piles-Theory.pdf 

    Args:
        pile_D (pytensor array): pile diameter profile with depth
        layer_type (pytensor array): layer type profile with depth. 0 for clay, 1 for sand
        c1 (pytensor array): for each layer, the end bearing capacity factor: either N_c (clay) or N_q (sand)
        c2 (pytensor array): for each layer, either beta (sand) or psi (clay)
    """

    # Geometric constants
    pile_A = np.pi * pile_D**2 / 4 # size N=100 to match forces
    pile_C = np.pi * pile_D

    # Create coordinate system
    zs = np.linspace(0, pile_L, N)
    dz = pile_L / (N-1) # length of one element

    # Generate soil property profiles with depth (each layer is uniform)
    # CAN REPLACE THE TOP PART WITH np.searchsorted()!!
    idxs = np.round(np.array(N * l_base_depth / pile_L)).astype(int)
    idxs = np.diff(idxs, prepend=0)
    gamma_d = np.repeat(l_gamma_d, idxs)#, total_repeat_length=N-1)
    e = np.repeat(l_e, idxs)#, total_repeat_length=N-1)
    c1 = np.repeat(l_c1, idxs)#, total_repeat_length=N-1)
    c2 = np.repeat(l_c2, idxs)#, total_repeat_length=N-1)
    shaft_pressure_limit = np.repeat(l_shaft_pressure_limit, idxs)#, total_repeat_length=N-1)
    layer_type = np.repeat(l_layer_type, idxs)#, total_repeat_length=N-1)
    end_pressure_limit = l_end_pressure_limit[-1] # only one value for end bearing limit
    gamma_sat = gamma_d + gamma_w * e / (1 + e)
    alpha = np.where(c2 <= 1.0, 0.5 * c2**(-0.5), 0.5 * c2**(-0.25)) # only used for clay

    # Generate effective vertical stress profile with depth
    eff_stress_increments = np.where(
        zs >= z_w,
        dz * (gamma_sat - gamma_w), # above water table
        dz * gamma_d) # below water table
    eff_stress = np.cumsum(eff_stress_increments)

    # gamma_sat cannot be less than gamma_w
    too_light = False
    if np.any(l_gamma_d < gamma_w): too_light = True

    # Generate tau_ult profile with depth
    tau_ult = np.where(
        layer_type == 0, # 0 for clay, 1 for sand
        alpha * c2 * eff_stress, # clay
        c2 * eff_stress) # sand
    tau_ult_over_AE = tau_ult / (pile_A * pile_E) # size N=100 to match forces
    
    # Get end bearing capacity
    Q_ult = pile_A[-1] * np.where(
        layer_type[-1] == 0, # 0 for clay, 1 for sand
        c1[-1] * c2[-1] * eff_stress[-1], # clay
        c1[-1] * eff_stress[-1]) # sand
    Q_ult_over_AE = Q_ult / (pile_A[-1] * pile_E)
    
    # Get shear force and end bearing force limits
    Q_limit = pile_A[-1] * end_pressure_limit
    Q_limit_over_AE = Q_limit / (pile_A[-1] * pile_E)
    S_limit = pile_C * dz * shaft_pressure_limit
    
    # Check if ultimate soil capacity is exceeded, if so return jax.nan to mark the parameter combination as invalid
    # NOTE! clay can lose capacity with high displacement, so this check needs to be refined for clay.
    # Really I will also check if the solver fails, and ignore it if it does. I should track both types of failure and report them after inference.
    Q_cap = np.minimum(Q_limit, Q_ult)
    S_cap = np.minimum(pile_C * dz * tau_ult, S_limit)
    P_cap = Q_cap + S_cap.sum()

    if P <= 0.99*P_cap and not too_light:

        P_over_AE = P / (pile_A[0] * pile_E) # initial strain at the head of the pile
        def f_bc(ya, yb):
            # Axial force in pile at the head must equal P
            strain_head = -ya[1]
            condition_head = P_over_AE - strain_head

            # Axial force in pile at the base must equal Q
            strain_base = -yb[1]
            Q_over_AE_base = np.minimum(Q_over_Q_ult(yb[0] / pile_D[-1]) * Q_ult_over_AE, Q_limit_over_AE)
            condition_base = Q_over_AE_base - strain_base

            return np.array([condition_head, condition_base])
        
        args=(zs, pile_A, pile_C, pile_E, pile_D, l_layer_type, l_base_depth, t_res_clay, shaft_pressure_limit, tau_ult_over_AE)
        def f_rk4(z, y):
            """Finds f such that dy/dz = f(y, z)
            In this case, y = [u, dudz] where s is scaling factor to improve numerical accuracy.
            """

            return np.array([
                y[1],
                f_acceleration(z, y[0], *args)
            ])
        
        # print(f"Args: {args}")
        # # # print out all the input parameters for debugging
        # print("pile_D:", pile_D)
        # print("pile_L:", pile_L)
        # print("pile_E:", pile_E)
        # print("l_layer_type:", l_layer_type)
        # print("l_gamma_d:", l_gamma_d)
        # print("l_e:", l_e)
        # print("l_c1:", l_c1)
        # print("l_c2:", l_c2)
        # print("l_shaft_pressure_limit:", l_shaft_pressure_limit)
        # print("l_end_pressure_limit:", l_end_pressure_limit)
        # print("l_base_depth:", l_base_depth)
        # print("P:", P)
        # print("z_w:", z_w)
        # print("N:", N)
        # print("t_res_clay:", t_res_clay)
        # print("tol:", tol)
        # print("do_print:", do_print)

        y0 = np.zeros(shape=(2, N))
        bvp_result = scipy.integrate.solve_bvp(f_rk4, f_bc, zs, y0, tol=tol, max_nodes=max_nodes)
        y_profile = bvp_result.sol(zs)

        u = y_profile[0]
        strain = - y_profile[1]
        force = pile_A * pile_E * strain

        # Find Q at the base of the pile, according to the displacement profile.
        # Then, use that to find zero_equilibrium, which should be zero at equilibrium.
        Q = np.minimum(Q_limit, Q_ult * Q_over_Q_ult(u[-1] / pile_D[-1]))
        zero_equilibrium = Q - force[-1]

        # convergence criterion to ensure the differential equation is satisfied
        acc_de = f_acceleration(zs, u, zs, pile_A, pile_C, pile_E, pile_D, l_layer_type, l_base_depth, t_res_clay, shaft_pressure_limit, tau_ult_over_AE)[1:-1]
        acc_diff = second_order_central_difference(u, dz)
        l2loss = np.sqrt(np.sum((acc_de - acc_diff)**2)) / np.sqrt(np.sum(acc_de**2))

        if bvp_result.status != 0 and bvp_result.status != 1:
            # status = 0 means the solver converged.
            # status = 1 means the maximum number of mesh nodes was exceeded. the solution can still be valid, the DE checks below can confim a good solution.

            print("Solver did not converge, status:", bvp_result.status, "message:", bvp_result.message)
            print(f"l_gamma_d: {l_gamma_d}, l_e: {l_e}, l_c2: {l_c2}, l_shaft_pressure_limit: {l_shaft_pressure_limit}, P/P_cap: {P / P_cap:.4e}")
            print(f"l2loss: {l2loss:.4e}, zero_eq: {zero_equilibrium:.4e}, zero_eq/P: {abs(zero_equilibrium / P):.4e}, bc: {f_bc(y_profile[:, 0], y_profile[:, -1])}")
            print(f"P: {P:.4e}, P_cap: {P_cap:.4e}")

        # DE checks to confirm a good solution.
        if l2loss > 0.1 or abs(zero_equilibrium / P) > 1e-2:
            logstr = f"WARNING: solve_bvp did not fulfil convergence criteria. l2loss: {l2loss:.4e}, zero_eq: {zero_equilibrium:.4e}, zero_eq/P: {abs(zero_equilibrium / P):.4e}, bc: {f_bc(y_profile[:, 0], y_profile[:, -1])}"
            logger.warning(logstr)
            print(logstr)
            print(f"l_gamma_d: {l_gamma_d}, l_e: {l_e}, l_c2: {l_c2}, l_shaft_pressure_limit: {l_shaft_pressure_limit}, P/P_cap: {P / P_cap:.4e}")
            force = np.full_like(force, np.nan)
        else:
            if do_print:
                logstr = f"solve_bvp converged. l2loss: {l2loss:.4e}, zero_eq: {zero_equilibrium:.4e}, zero_eq/P: {abs(zero_equilibrium / P):.4e}"
                logger.info(logstr)
                print(logstr)

    else:
        if do_print:
            print("Returning P > 0.99*P_cap OR too_light")

        u = np.full_like(zs, np.nan)
        strain = np.full_like(zs, np.nan)
        force = np.full_like(zs, np.nan)
        zero_equilibrium = np.nan
        l2loss = np.nan
        Q = np.nan

    # debugging
    # extra = None
    # if data is not None:
    #     e1 = f_acceleration(zs, data, zs, pile_A, pile_C, pile_E, pile_D, l_layer_type, l_base_depth, t_res_clay, shaft_pressure_limit, tau_ult_over_AE)[1:-1]
    #     e2 = second_order_central_difference(data, dz)
    #     extra = np.sqrt(np.sum((e1 - e2)**2)) / np.sqrt(np.sum(e1**2))

    tau = np.where(
        layer_type == 0, # 0 for clay, 1 for sand
        tau_over_tau_ult_clay(u / pile_D, t_res=t_res_clay),
        tau_over_tau_ult_sand(u / pile_D)
    )

    return SolveData(force, strain, u, zero_equilibrium, l2loss, tau, Q, shaft_pressure_limit,
                     Q_limit, eff_stress, tau_ult, Q_ult, P, P_cap, Q_cap, S_cap, too_light)
    # return u, strain, force, tau_ult_over_AE, extra, zero_equilibrium, P_cap, l2loss
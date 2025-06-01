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

def f_acceleration(z, u, m, zs, pile_A, pile_C, pile_E, pile_D, l_layer_type, l_base_depth, t_res_clay, shaft_pressure_limit, tau_ult_over_AEm):
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
            tau_over_tau_ult_clay(u*m / D, t_res=t_res_clay),
            tau_over_tau_ult_sand(u*m / D)
        )
    except IndexError as e:
        print(z, l_base_depth, layer_ids)
        raise IndexError from e
    tau_over_AEm = np.clip(tau_over_AEm, -shaft_pressure_limit/(A*pile_E*m), shaft_pressure_limit/(A*pile_E*m))

    return C * tau_over_AEm

def f_rk4(z, y, *args):
    """Finds f such that dy/dz = f(y, z)
    In this case, y = [u, dudz] / m where s is scaling factor to improve numerical accuracy.
    """

    return np.array([
        y[1],
        f_acceleration(z, y[0], *args)
    ])

class SolveData():
    def __init__(self, F, strain, u, zero_equilibrium, l2loss, tau, Q,
                 shaft_pressure_limit, Q_limit, eff_stress, tau_ult, Q_ult, P, P_cap,
                 Q_cap=None, S_cap=None):
        
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

def forward_api(pile_D, pile_L, pile_E, # pile parameters
                l_layer_type, l_gamma_d, l_e, l_c1, l_c2, l_shaft_pressure_limit, l_end_pressure_limit, l_base_depth, # soil parameters.
                P, z_w, N=100, t_res_clay=0.9,
                u0 = 0.0, m = None, data = None, rtol=1e-10, atol=1e-10):
    """Implementation of RSPile axially loaded 1D FEM pile/soil stress/strain model using spring discretisation.
    tau_ult and q_ult calculations for sands and clays are taken from API2GEO 2011 (Reading 8.2).
    https://static.rocscience.cloud/assets/verification-and-theory/RSPile/RSPile-Axially-Loaded-Piles-Theory.pdf 

    Args:
        pile_D (pytensor array): pile diameter profile with depth
        layer_type (pytensor array): layer type profile with depth. 0 for clay, 1 for sand
        c1 (pytensor array): for each layer, the end bearing capacity factor: either N_c (clay) or N_q (sand)
        c2 (pytensor array): for each layer, either beta (sand) or psi (clay)
    """

    if m is None: m = 1
    # print(f"u0: {u0}")

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
    
    # Get shear force and end bearing force limits
    Q_limit = pile_A[-1] * end_pressure_limit
    S_limit = pile_C * dz * shaft_pressure_limit
    
    # Check if ultimate soil capacity is exceeded, if so return jax.nan to mark the parameter combination as invalid
    # NOTE! clay can lose capacity with high displacement, so this check needs to be refined for clay.
    # Really I will also check if the solver fails, and ignore it if it does. I should track both types of failure and report them after inference.
    Q_cap = np.minimum(Q_limit, Q_ult)
    S_cap = np.minimum(pile_C * dz * tau_ult, S_limit)
    P_cap = Q_cap + S_cap.sum()

    if P <= P_cap:
        dudz0 = - P / (pile_A[0] * pile_E) # initial strain
        y_init = np.array([u0, dudz0])/m

        y_profile = scipy.integrate.solve_ivp(f_rk4, (0, pile_L), y_init, t_eval=zs, method="Radau", rtol=rtol, atol=atol, first_step=dz,
                                            args=(m, zs, pile_A, pile_C, pile_E, pile_D, l_layer_type, l_base_depth, t_res_clay, shaft_pressure_limit, tau_ult_over_AE/m))

        u = y_profile.y[0] * m
        strain = - y_profile.y[1] * m
        force = pile_A * pile_E * strain

        # Find Q at the base of the pile, according to the displacement profile.
        # Then, use that to find zero_equilibrium, which should be zero at equilibrium.
        Q = np.minimum(Q_limit, Q_ult * Q_over_Q_ult(u[-1] / pile_D[-1]))
        zero_equilibrium = Q - force[-1]

        # convergence criterion to ensure the differential equation is satisfied
        acc_de = m * f_acceleration(zs, u/m, m, zs, pile_A, pile_C, pile_E, pile_D, l_layer_type, l_base_depth, t_res_clay, shaft_pressure_limit, tau_ult_over_AE/m)[1:-1]
        acc_diff = second_order_central_difference(u, dz)
        l2loss = np.sqrt(np.sum((acc_de - acc_diff)**2)) / np.sqrt(np.sum(acc_de**2))

    else:
        u = np.full_like(zs, np.nan)
        strain = np.full_like(zs, np.nan)
        force = np.full_like(zs, np.nan)
        zero_equilibrium = np.nan
        l2loss = np.nan
        Q = np.nan

    # debugging
    extra = None
    if data is not None:
        e1 = m * f_acceleration(zs, data/m, m, zs, pile_A, pile_C, pile_E, pile_D, l_layer_type, l_base_depth, t_res_clay, shaft_pressure_limit, tau_ult_over_AE/m)[1:-1]
        e2 = second_order_central_difference(data, dz)
        extra = np.sqrt(np.sum((e1 - e2)**2)) / np.sqrt(np.sum(e1**2))

    tau = np.where(
        layer_type == 0, # 0 for clay, 1 for sand
        tau_over_tau_ult_clay(u / pile_D, t_res=t_res_clay),
        tau_over_tau_ult_sand(u / pile_D)
    )

    return SolveData(force, strain, u, zero_equilibrium, l2loss, tau, Q, shaft_pressure_limit,
                     Q_limit, eff_stress, tau_ult, Q_ult, P, P_cap, Q_cap, S_cap)
    # return u, strain, force, tau_ult_over_AE, extra, zero_equilibrium, P_cap, l2loss

def shooting_api(u0_init, **forward_params):
    # print("Solving shooting. Warning; no xtol and rtol set.")

    def forward_wrapper(u0, **forward_params):
        return forward_api(u0=u0, **forward_params).zero_equilibrium
    def closure(u0):
        # print(f"guessing {u0:.6e}")
        return forward_wrapper(u0, **forward_params)
    
    # Check if P > P_cap
    result_check = forward_api(u0=u0_init, **forward_params)
    if forward_params["P"] > result_check.P_cap:
        print("returning P > P_cap")
        return result_check
    
    # limit result to +-10cm
    # note, when bracket is used, x0 is actually ignored.
    result = scipy.optimize.root_scalar(closure, x0=u0_init, bracket=[-0.1, 0.1])
    if not result.converged:
        logstr = f"WARNING: shooting did not converge. Message: {result.flag}"
        logger.warning(logstr)
        print(logstr)
    
    u0_solved = result.root
    ret = forward_api(u0=u0_solved, **forward_params)

    # check the convergence criteria
    # zero_equilibrium is worrying if it's of the same order as the true end bearing FORCE capacity (~10^6).
    zero_eq = ret.zero_equilibrium
    l2loss = ret.l2loss
    if l2loss > 0.1 or abs(zero_eq / forward_params["P"]) > 1e-2:
        logstr = f"WARNING: shooting did not fulfil convergence criteria. l2loss: {l2loss:.4e}, zero_eq: {zero_eq:.4e}, zero_eq/P: {zero_eq / forward_params['P']:.4e}"
        logger.warning(logstr)
        print(logstr)
        print(f"l_gamma_d: {forward_params["l_gamma_d"]}")
        ret.force = np.full_like(ret.force, np.nan)

        # usr_inp = input("save to csv? (filename/n)")
        # if usr_inp != "n":

        #     dict_in = forward_params
        #     utility.numpy_dict_to_json(dict_in, f'{usr_inp}_in.json')

        #     dict_out = {
        #         "u": ret[0],
        #         "strain": ret[1],
        #         "forces": ret[2],
        #         "P_cap": ret[6],
        #         "l2loss": l2loss,
        #         "zero_eq": zero_eq,
        #     }
        #     utility.numpy_dict_to_json(dict_out, f'{usr_inp}_out.json')

        #     print("saved!")

    else:
        logstr = f"shooting converged. l2loss: {l2loss:.4e}, zero_eq: {zero_eq:.4e}, zero_eq/P: {zero_eq / forward_params['P']:.4e}"
        logger.info(logstr)
        print(logstr)

    return ret

def shooting_api_wrapper(*args):
    argnames = ["pile_D", "pile_L", "pile_E", "l_layer_type", "l_gamma_d", "l_e", "l_c1", "l_c2",
                "l_shaft_pressure_limit", "l_end_pressure_limit", "l_base_depth", # soil parameters.
                "P", "z_w", "N", "t_res_clay"]
    
    kwargs = {k: v for k, v in zip(argnames, args)}

    logger.debug(f"Shooting API, l_gamma_d: {kwargs['l_gamma_d']}")

    return shooting_api(u0_init=0, **kwargs)
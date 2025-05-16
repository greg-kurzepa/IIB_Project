import numpy as np
import scipy
import scipy.optimize

m_to_in = 39.3701
gamma_w = 9.81e3 # unit weight of water

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

def f_simultaneous_nondim(x, dz, N, pile, P_over_AEs, Q_ult_over_AEs, tau_ult_over_AEs, Q_over_Q_ult_func, tau_over_tau_ult_func, tau_limit_over_AEs, Q_limit_over_AEs):
    d = x[:N]
    u = x[N:]

    dz_halfel = dz/2
    strain_top = (d[:-1] - u) / dz_halfel
    strain_bottom = (u - d[1:]) / dz_halfel

    F_over_AEs_tip = min(Q_ult_over_AEs * Q_over_Q_ult_func(d[-1]), Q_limit_over_AEs)
    F_over_AEs_top_excluding_tip = pile.F_over_AEs(strain_top, d[:-1])
    F_over_AEs_top = np.append(F_over_AEs_top_excluding_tip, F_over_AEs_tip)
    
    F_over_AEs_bottom_excluding_head = pile.F_over_AEs(strain_bottom, d[1:])
    F_over_AEs_bottom = np.insert(F_over_AEs_bottom_excluding_head, 0, P_over_AEs)

    S_over_AEs = pile.C * dz * np.clip(tau_ult_over_AEs * tau_over_tau_ult_func(u), -tau_limit_over_AEs, tau_limit_over_AEs)

    zeros_1 = F_over_AEs_bottom - F_over_AEs_top
    zeros_2 = F_over_AEs_top[1:] + S_over_AEs - F_over_AEs_bottom[:-1]

    return np.concatenate((zeros_1, zeros_2))

def solve_springs4(pile, soil, P, z_w, N=100, t_res_clay=0.9,
                   tau_over_tau_ult_func = None, Q_over_Q_ult_func = None,
                   tol=1e-8, outtol=1e-2):
    """Implementation of RSPile axially loaded 1D FEM pile/soil stress/strain model using spring discretisation.
    tau_ult and q_ult calculations for sands and clays are taken from API2GEO 2011 (Reading 8.2).
    https://static.rocscience.cloud/assets/verification-and-theory/RSPile/RSPile-Axially-Loaded-Piles-Theory.pdf 

    Here I use a different set of equilibrium equations (more general) compared to solve_springs3, so that:
    - I can have an arbitrary stress-strain relationship for the concrete elements (used to be uniformly elastic)
    - I can have an arbitrary pile cross-sectional alrea profile with depth (used to be constant)

    Args:
        pile (_pile_and_soil.Pile): the pile object
        soil (_pile_and_soil.Soil): the soil object containing the soil layers
        P (float): the axial load at the top of the pile
        z_w (float): the water table depth
        N (int, optional): the number of nodes along the pile. Defaults to 100.
        tol (float, optional): the tolerance for the solver. Defaults to 1e-8.
    """

    # for idx, layer in enumerate(soil.layers):
    #     print(f"layer {idx}: {layer}")
    # print(f"pile: {pile}")
    # print(f"P: {P}, z_w: {z_w}, N: {N}, t_res_clay: {t_res_clay}")

    # pile cross-sectional area profile with depth (uniform for now)
    A = np.full(N, pile.A)
    A_midpoints = (A[:1] + A[:-1]) / 2

    # Create coordinate system
    z = np.linspace(0, pile.L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:]) # coordinates of soil element nodes
    dz = pile.L / (N-1) # length of one element
    dz_halfel = 0.5 * dz # half the length of one element
    
    # For each soil node, get the soil layer and soil type it is in.
    layer_ids = np.array([soil.find_layer_id_at_depth(z_i) for z_i in z_midpoints])
    layer_types = np.array([soil.layers[layer_ids[i]].layer_type for i in range(N-1)])
    # Sands have a limit on the shaft friction and end bearing force. For layers without a limit it's just np.inf
    shaft_pressure_limit = np.array([soil.layers[layer_ids[i]].shaft_pressure_limit for i in range(N-1)])
    Q_limit = A[-1] * soil.layers[-1].end_pressure_limit

    # generate effective vertical stress profile with depth
    eff_stress_increments = np.array([
        soil.layers[layer_ids[i]].gamma_d * dz if z_midpoints[i] <= z_w else
        (soil.layers[layer_ids[i]].gamma_sat - gamma_w) * dz
    for i in range(N-1)])
    eff_stress = np.cumsum(eff_stress_increments)

    # generate tau_ult profile with depth
    tau_ult = np.array([soil.layers[layer_ids[i]].tau_ult(eff_stress[i]) for i in range(N-1)])

    # get end bearing capacity
    Q_ult = A[-1] * soil.layers[-1].q_ult(eff_stress[-1])
    
    # Check if ultimate soil capacity is exceeded, if so return jax.nan to mark the parameter combination as invalid
    # NOTE! clay can lose capacity with high displacement, so this check needs to be refined for clay.
    # Really I will also check if the solver fails, and ignore it if it does. I should track both types of failure and report them after inference.
    Q_cap = np.minimum(Q_ult, Q_limit)
    S_cap = pile.C * dz * np.minimum(tau_ult, shaft_pressure_limit)
    P_cap = Q_cap + S_cap.sum()

    # allows for custom constitutive functions for the soil, e.g. purely elastic for testing against Pulous results
    # this is the default, which conforms to API2GEO
    if tau_over_tau_ult_func is None:
        tau_over_tau_ult_func = lambda u : np.where(layer_types == "clay", tau_over_tau_ult_clay(u / pile.D, t_res=t_res_clay), tau_over_tau_ult_sand(u / pile.D))
    if Q_over_Q_ult_func is None:
        Q_over_Q_ult_func = lambda d_tip : Q_over_Q_ult(d_tip / pile.D)

    if P <= P_cap:
        # initial guesses
        # u is displacement of an element midpoints, d is displacement at nodes (i.e. element edges)
        d_initial = np.zeros_like(z)
        u_initial = np.zeros_like(z_midpoints) # d for a element N is the displacement at the bottom of the element, i.e. the pile head displacement is not included
        initial = np.concatenate((d_initial, u_initial))
        
        nondim_args = {"dz" : dz, "N" : N, "pile" : pile, "P_over_AEs" : P/(A[0]*pile.steel_E), "Q_ult_over_AEs" : Q_ult/(A[-1]*pile.steel_E),
                        "tau_ult_over_AEs" : tau_ult/(A_midpoints*pile.steel_E), "Q_over_Q_ult_func" : Q_over_Q_ult_func,
                        "tau_over_tau_ult_func" : tau_over_tau_ult_func, "tau_limit_over_AEs" : shaft_pressure_limit/(A_midpoints*pile.steel_E),
                        "Q_limit_over_AEs" : Q_limit/(A[-1]*pile.steel_E)}

        # res, infodict, ier, mesg = scipy.optimize.fsolve(f_simultaneous_nondim, initial, xtol=tol, full_output=True, args=tuple(nondim_args.values()))
        obj = scipy.optimize.root(f_simultaneous_nondim, initial, method="hybr", tol=tol, args=tuple(nondim_args.values()))
        res, ier, mesg = obj.x, obj.success, obj.message

        zeros = f_simultaneous_nondim(res, **nondim_args)

        # check if the solver converged
        if any(abs(zeros) > outtol):
            print(f"Warning: Absolute fsolve error was greater than outtol. outtol is {outtol:.4e}, max error was {np.abs(zeros).max():.4e} ier: {ier}, mesg: {mesg}")

    else:
        # If the ultimate capacity is exceeded, return NaN
        # This allows the pymc op to treat it as zero-probability
        res = np.full(2*N-1, np.nan)
        zeros = np.full(2*N-1, np.nan)

    d = res[:N]
    u = res[N:]

    strain_top = (d[:-1] - u) / dz_halfel
    strain_tip = (u[-1] - d[-1]) / dz_halfel
    strain = np.append(strain_top, strain_tip)

    F = A * pile.steel_E * pile.F_over_AEs(strain, d) # sanity check, should equal F (it did)
    tau = np.clip(tau_ult * tau_over_tau_ult_func(u), -shaft_pressure_limit, shaft_pressure_limit)
    Q = min(Q_ult * Q_over_Q_ult_func(d[-1]), Q_limit)

    return F, strain, d, u, zeros, tau, Q, eff_stress, P, P_cap
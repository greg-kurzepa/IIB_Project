import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy
import scipy.optimize

m_to_in = 39.3701
gamma_w = 9.81e3 # unit weight of water

def tau_over_tau_ult_clay(disp_over_D):
    # values to interpolate from
    z_over_D = np.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, np.inf]) / m_to_in
    tau_over_tau_ult = np.array([0, 0.3, 0.5, 0.75, 0.9, 1, 0.9, 0.9])

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
    return disp_over_D*np.interp(disp_over_D, z_over_D, Q_over_Q_ult)

def solve_springs(pile, soil, P, N=100, tol=1e-8):
    """Implementation of RSPile axially loaded 1D FEM pile/soil stress/strain model using spring discretisation.
    The springs are linear (elastic) for the pile and nonlinear for the soil.
    https://static.rocscience.cloud/assets/verification-and-theory/RSPile/RSPile-Axially-Loaded-Piles-Theory.pdf 

    Args:
        pile (pytensor): the tensor of pile properties. dimensions (n_pile_properties).
        soil (pytensor): the tensor of soil properties. dimensions (n_layers x n_soil_properties).
        z (pytensor): depths array. dimensions (n_nodes).
        P (float): the axial load at the top of the pile
        N (int, optional): the number of nodes along the pile. Defaults to 100.
        tol (float, optional): the tolerance for the solver. Defaults to 1e-8.
    """

    # Create coordinate system
    z = np.linspace(0, pile.L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:]) # coordinates of soil element nodes
    dz = pile.L / (N-1) # length of one element

    # Define intermediate constants
    s_u_base = get_undrained_strength(soil, pile.L) # undrained strength at base
    total_weight = s_u_base - soil.layers[-1].s_u0 # total weight of soil above base of pile
    Q_ult = np.pi * pile.A * (soil.layers[-1].N_c * s_u_base) # ultimate end bearing pressure
    k = 2 * pile.A * pile.E / dz

    # For each soil node, assign the soil layer it is in
    midpoints_layer_ids = [soil.find_layer_id_at_depth(z_i) for z_i in z_midpoints]

    # Define lists of parameters at each soil node
    # (instead of this, could do find_layer_id_at_depth each time but I reckon that would be slower)
    alpha = np.array([soil.layers[layer_id].alpha for layer_id in midpoints_layer_ids])
    gamma = np.array([soil.layers[layer_id].gamma for layer_id in midpoints_layer_ids])
    N_c = np.array([soil.layers[layer_id].N_c for layer_id in midpoints_layer_ids])
    s_u = vectorised_get_undrained_strength(soil, z_midpoints)
    rho = np.array([soil.layers[layer_id].rho for layer_id in midpoints_layer_ids])
    base_depth = np.array([soil.layers[layer_id].base_depth for layer_id in midpoints_layer_ids])
    tau_ult = alpha * s_u # ultimate skin friction

    # initial guess is zeros
    u_guess = np.zeros_like(z_midpoints)

    def f_simultaneous(u):
        # get LHS of simultaneous equations
        lhs = 1 - np.cumsum(pile.C * dz * tau_ult * tau_over_tau_ult(u / pile.D) / P)

        # get RHS of simultaneous equations
        disp_diff = u[:-1] - u[1:]
        d_N = scipy.optimize.fsolve(
            lambda d_N : k*(u[-1] - d_N) - Q_ult * Q_over_Q_ult(d_N / pile.D),
            u[-1] # starting guess
        )
        rhs = np.append(k / (2*P) * disp_diff, k / P * (u[-1] - d_N))

        # returns vector size N-1 = size(mu)
        return lhs - rhs
    
    u = scipy.optimize.fsolve(f_simultaneous, u_guess, xtol=tol)
    tau_ult = alpha * s_u # ultimate skin friction
    shear = pile.C * dz * tau_ult * tau_over_tau_ult(u / pile.D)
    cumulative_shear_from_base = np.cumsum(np.insert(shear, 0, 0))
    F = P - cumulative_shear_from_base
    strain = F / (pile.A * pile.E)
    d = np.insert(u - F[1:] / k, 0, u[0] + P / k) # pile node displacements
    # TODO: as a sanity check, implement comparison of u from e1 and e2 :)

    return F, strain, d

def solve_springs2(pile, soil, P, z_w, N=100, tol=1e-8):
    # !!! CURRENTLY FOR SAND ONLY (TAU_OVER_TAU_ULT_SAND)

    """Implementation of RSPile axially loaded 1D FEM pile/soil stress/strain model using spring discretisation.
    tau_ult and q_ult calculations for sands and clays are taken from API2GEO 2011 (Reading 8.2).
    https://static.rocscience.cloud/assets/verification-and-theory/RSPile/RSPile-Axially-Loaded-Piles-Theory.pdf 

    Args:
        pile (_pile_and_soil.Pile): the pile object
        soil (_pile_and_soil.Soil): the soil object containing the soil layers
        P (float): the axial load at the top of the pile
        z_w (float): the water table depth
        N (int, optional): the number of nodes along the pile. Defaults to 100.
        tol (float, optional): the tolerance for the solver. Defaults to 1e-8.
    """

    # Create coordinate system
    z = np.linspace(0, pile.L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:]) # coordinates of soil element nodes
    dz = pile.L / (N-1) # length of one element
    
    # For each soil node, assign the soil layer it is in
    layer_ids = [soil.find_layer_id_at_depth(z_i) for z_i in z_midpoints]

    # generate effective vertical stress profile with depth
    eff_stress_increments = np.array([
        soil.layers[layer_ids[i]].gamma_d * dz if z_midpoints[i] <= z_w else
        (soil.layers[layer_ids[i]].gamma_sat - gamma_w) * dz
    for i in range(N-1)])
    eff_stress = np.cumsum(eff_stress_increments)

    # generate tau_ult profile with depth
    tau_ult = np.array([soil.layers[layer_ids[i]].tau_ult(eff_stress[i]) for i in range(N-1)])

    # get end bearing capacity
    Q_ult = pile.A * soil.layers[-1].q_ult(eff_stress[-1])

    # pile spring constant for one element
    k = 2 * pile.A * pile.E / dz

    # initial guess is zeros
    u_guess = np.zeros_like(z_midpoints)

    def f_simultaneous(u):
        # get LHS of simultaneous equations
        lhs = 1 - np.cumsum(pile.C * dz * tau_ult * tau_over_tau_ult_sand(u / pile.D) / P)

        # get RHS of simultaneous equations
        disp_diff = u[:-1] - u[1:]
        d_N = scipy.optimize.fsolve(
            lambda d_N : k*(u[-1] - d_N) - Q_ult * Q_over_Q_ult(d_N / pile.D),
            u[-1] # starting guess
        )
        rhs = np.append(k / (2*P) * disp_diff, k / P * (u[-1] - d_N))

        # returns vector size N-1 = size(mu)
        return lhs - rhs
    
    u = scipy.optimize.fsolve(f_simultaneous, u_guess, xtol=tol)
    shear = pile.C * dz * tau_ult * tau_over_tau_ult_sand(u / pile.D)
    cumulative_shear_from_base = np.cumsum(np.insert(shear, 0, 0))
    F = P - cumulative_shear_from_base
    strain = F / (pile.A * pile.E)
    d = np.insert(u - F[1:] / k, 0, u[0] + P / k) # pile node displacements
    # TODO: as a sanity check, implement comparison of u from e1 and e2 :)

    return F, strain, d

def solve_springs3(pile, soil, P, z_w, N=100, tol=1e-8):
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

    # Create coordinate system
    z = np.linspace(0, pile.L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:]) # coordinates of soil element nodes
    dz = pile.L / (N-1) # length of one element
    dz_halfel = 0.5 * dz # half the length of one element
    
    # For each soil node, assign the soil layer it is in
    layer_ids = [soil.find_layer_id_at_depth(z_i) for z_i in z_midpoints]

    # generate effective vertical stress profile with depth
    eff_stress_increments = np.array([
        soil.layers[layer_ids[i]].gamma_d * dz if z_midpoints[i] <= z_w else
        (soil.layers[layer_ids[i]].gamma_sat - gamma_w) * dz
    for i in range(N-1)])
    eff_stress = np.cumsum(eff_stress_increments)

    # generate tau_ult profile with depth
    tau_ult = np.array([soil.layers[layer_ids[i]].tau_ult(eff_stress[i]) for i in range(N-1)])

    # pile cross-sectional area profile with depth (uniform for now)
    A = np.full(N, pile.A)

    # get end bearing capacity
    Q_ult = A[-1] * soil.layers[-1].q_ult(eff_stress[-1])

    # initial guesses
    # u is displacement of an element midpoints, d is displacement at nodes (i.e. element edges)
    u_initial = np.zeros_like(z_midpoints)
    d_initial = np.zeros_like(z_midpoints) # d for a element N is the displacement at the bottom of the element, i.e. the pile head displacement is not included
    initial = np.concatenate((u_initial, d_initial))

    def f_simultaneous(x):
        u = x[:N-1]
        d = x[N-1:]
        
        F_bottom = A[1:] * pile.stress_from_strain((u - d) / dz_halfel)

        F_top_excluding_tip = A[2:] * pile.stress_from_strain((d[:-1] - u[1:]) / dz_halfel)
        F_tip = Q_ult * Q_over_Q_ult(d[-1] / pile.D)
        F_top = np.append(F_top_excluding_tip, F_tip)

        S = pile.C * dz * tau_ult * tau_over_tau_ult_sand(u / pile.D)

        zeros_1 = F_bottom - F_top
        zeros_2 = F_top + S - np.insert(F_top[:-1], 0, P)

        return np.concatenate((zeros_1, zeros_2))
    
    res = scipy.optimize.fsolve(f_simultaneous, initial, xtol=tol)
    u = res[:N-1]
    d_excluding_head = res[N-1:]
    strain_head = pile.strain_from_stress_pick_lowest(P / A[0])
    d_head = strain_head * dz_halfel + u[0]
    d = np.insert(d_excluding_head, 0, d_head)

    strain_excluding_tip = (d[:-1] - u) / dz_halfel
    strain_tip = (u[-1] - d[-1]) / dz_halfel
    strain = np.append(strain_excluding_tip, strain_tip)

    shear = pile.C * dz * tau_ult * tau_over_tau_ult_sand(u / pile.D)
    cumulative_shear_from_top = np.cumsum(np.insert(shear, 0, 0))
    F = P - cumulative_shear_from_top
    # F2 = A * pile.stress_from_strain(strain) # sanity check, should equal F (it did)

    return F, strain, d
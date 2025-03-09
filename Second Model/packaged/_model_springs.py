import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy
import scipy.optimize

m_to_in = 39.3701

def tau_over_tau_ult(disp_over_D):
    # values to interpolate from
    z_over_D = np.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, np.inf]) / m_to_in
    tau_over_tau_ult = np.array([0, 0.3, 0.5, 0.75, 0.9, 1, 0.9, 0.9])

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

def solve_springs2(pile, soil, P, N=100, tol=1e-8):
    """Implementation of RSPile axially loaded 1D FEM pile/soil stress/strain model using spring discretisation.
    tau_ult and q_ult calculations for sands and clays are taken from API2GEO 2011 (Reading 8.2).
    https://static.rocscience.cloud/assets/verification-and-theory/RSPile/RSPile-Axially-Loaded-Piles-Theory.pdf 

    Regardless of soil or clay, the input parameters for the soil are:
    - alpha, such that tau_ult = alpha * sigma_v' (where sigma_v' is the effective vertical stress)
    - N_c, such that q_ult = N_c * sigma_v'.

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
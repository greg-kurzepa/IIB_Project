import jax
import jax.numpy as jnp
import jax.scipy.optimize

m_to_in = 39.3701

# jax rules:
# - pure functions only (inputs fully determine outputs, and no side effects)
# - no slicing with values unknown at compile-time
# - no classes (use dictionaries instead)
# - 

def tau_over_tau_ult(disp_over_D):
    # values to interpolate from
    z_over_D = jnp.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, jnp.inf]) / m_to_in
    tau_over_tau_ult = jnp.array([0, 0.3, 0.5, 0.75, 0.9, 1, 0.9, 0.9])

    # below works for positive AND negative displacement.
    return jnp.sign(disp_over_D)*jnp.interp(jnp.abs(disp_over_D), z_over_D, tau_over_tau_ult)

def Q_over_Q_ult(disp_over_D):
    # values to interpolate from
    z_over_D = jnp.array([-jnp.inf, 0, 0.002, 0.013, 0.042, 0.073, 0.1, jnp.inf]) / m_to_in
    Q_over_Q_ult = jnp.array([0, 0, 0.25, 0.5, 0.75, 0.9, 1, 1])

    # below works for only downwards (+ve) displacement since going up there is no resistance.
    return disp_over_D*jnp.interp(disp_over_D, z_over_D, Q_over_Q_ult)

ALPHA_IDX, GAMMA_IDX, N_C_IDX, S_U0_IDX, RHO_IDX, BASE_DEPTH_IDX = 0, 1, 2, 3, 4, 5
PILE_R_IDX, PILE_L_IDX, PILE_W_IDX, PILE_E_IDX = 0, 1, 2, 3
# store the parameters of the pile in one vector (pile).
# store parameters of all the layers combined in one 2d array (soil).
# BUT it needs to be vectorised to accept batches of parameters! for now I will use jnp.vectorize() but this is less efficient than directly supporting batches.
def solve_springs_jax(pile, soil, P, N=100, tol=1e-8):
    """Implementation of RSPile axially loaded 1D FEM pile/soil stress/strain model using spring discretisation.
    The springs are linear (elastic) for the pile and nonlinear for the soil.
    https://static.rocscience.cloud/assets/verification-and-theory/RSPile/RSPile-Axially-Loaded-Piles-Theory.pdf 

    Args:
        pile: vector of pile proprties, jnp.array([R, L, W, E])
        soil: array of soil layers and their properties, jnp.array([[soil0_alpha, soil0_gamma, soil0_N_c, soil0_s_u0, soil0_rho, soil0_base_depth],
                                                                    [soil1_alpha, soil1_gamma, soil1_N_c, soil1_s_u0, soil1_rho, soil1_base_depth],
                                                                    ...])
    """

    R = pile[PILE_R_IDX]
    L = pile[PILE_L_IDX]
    W = pile[PILE_W_IDX]
    E = pile[PILE_E_IDX]
    A = jnp.pi * R**2
    C = 2 * jnp.pi * R
    D = 2 * R

    # Create coordinate system
    z = jnp.linspace(0, L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:]) # coordinates of soil element nodes
    dz = L / (N-1) # length of one element

    # generate property-with-z(-midpoints) arrays from soil layers using jnp.repeat. the total_repeat_length argument (=N) must be specified!
    # chatgpt suggested interpolating soil parameters for the two nodes around a boundary. could try that if this causes problems
    boundary_indexes = jnp.round(soil[:, BASE_DEPTH_IDX]*(N-1)/pile[PILE_L_IDX]).astype(int) # array of index locations of each boundary layer
    boundary_repeats = jnp.diff(boundary_indexes, prepend=0) # array of number of indices each layer fills
    alpha = jnp.repeat(soil[:, ALPHA_IDX], boundary_repeats)
    gamma = jnp.repeat(soil[:, GAMMA_IDX], boundary_repeats)
    N_c = jnp.repeat(soil[:, N_C_IDX], boundary_repeats)
    s_u0 = jnp.repeat(soil[:, S_U0_IDX], boundary_repeats)
    rho = jnp.repeat(soil[:, RHO_IDX], boundary_repeats)

    # Define ultimate skin friction and undrained shear strength with depth
    weight_profile = jnp.cumsum(rho * dz)
    s_u = s_u0 + weight_profile
    tau_ult = alpha * s_u

    # Define intermediate constants
    Q_ult = jnp.pi * A * (N_c[-1] * s_u[-1])
    k = 2 * A * E / dz

    # initial guess is zeros
    u_guess = jnp.zeros_like(z_midpoints)

    def f_simultaneous(u):
        # get LHS of simultaneous equations
        lhs = 1 - jnp.cumsum(C * dz * tau_ult * tau_over_tau_ult(u / D) / P)

        # get RHS of simultaneous equations
        disp_diff = u[:-1] - u[1:]

        # d_N = jax.scipy.optimize.minimize(
        #     lambda d_N : jnp.abs( k*(u[-1] - d_N) - Q_ult * Q_over_Q_ult(d_N / D) )[0],
        #     jnp.array([u[-1]]), # starting guess
        #     method="BFGS"
        # ).fun
        d_N = 0
        # print(type(k / (2*P) * disp_diff))
        # print(type(k), type(P), type(u[-1]), type(d_N))
        # print(u[-1], d_N)
        # print(type(k / P * (u[-1] - d_N)))
        rhs = jnp.append(k / (2*P) * disp_diff, k / P * (u[-1] - d_N))

        # returns (sum of) vector size N-1 = size(mu)
        return jnp.sum(jnp.abs(lhs - rhs)) # abs to make it always positive; we want to find the roots using the minimize function
    
    u = jax.scipy.optimize.minimize(f_simultaneous, u_guess, tol=tol, method="BFGS")
    tau_ult = alpha * s_u # ultimate skin friction
    shear = C * dz * tau_ult * tau_over_tau_ult(u / D)
    cumulative_shear_from_base = jnp.cumsum(jnp.insert(shear, 0, 0))
    F = P - cumulative_shear_from_base
    strain = F / (A * E)
    d = jnp.insert(u - F[1:] / k, 0, u[0] + P / k) # pile node displacements
    # TODO: as a sanity check, implement comparison of u from e1 and e2 :)

    return F, strain, d
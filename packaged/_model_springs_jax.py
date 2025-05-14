import jax
import jax.numpy as jnp
import jaxopt
import optimistix as optx
import optax

import json

m_to_in = 39.3701
gamma_w = 9.81e3 # unit weight of water
jax.config.update('jax_enable_x64', True) #ESSENTIAL

def tau_over_tau_ult_clay_api(disp_over_D, t_res=0.9):
    # values to interpolate from
    z_over_D = jnp.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, jnp.inf]) / m_to_in
    tau_over_tau_ult = jnp.array([0, 0.3, 0.5, 0.75, 0.9, 1, t_res, t_res])

    # below works for positive AND negative displacement.
    return jnp.sign(disp_over_D)*jnp.interp(jnp.abs(disp_over_D), z_over_D, tau_over_tau_ult)

def tau_over_tau_ult_sand_api(disp_over_D):
    # values to interpolate from
    z_over_D = jnp.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, jnp.inf]) / m_to_in
    tau_over_tau_ult = jnp.array([0, 0.3, 0.5, 0.75, 0.9, 1, 1, 1])

    # below works for positive AND negative displacement.
    return jnp.sign(disp_over_D)*jnp.interp(jnp.abs(disp_over_D), z_over_D, tau_over_tau_ult)

def Q_over_Q_ult_api(disp_over_D):
    # values to interpolate from
    z_over_D = jnp.array([-jnp.inf, 0, 0.002, 0.013, 0.042, 0.073, 0.1, jnp.inf]) / m_to_in
    Q_over_Q_ult = jnp.array([0, 0, 0.25, 0.5, 0.75, 0.9, 1, 1])

    # below works for only downwards (+ve) displacement since going up there is no resistance.
    return jnp.interp(disp_over_D, z_over_D, Q_over_Q_ult)

def pile_stress_from_strain(strain, E_ci, f_ctm, stress_ct, strain_ct):
    pass

def f_simultaneous_api_nondim(x, dz, N, t_res_clay, pile_D, pile_D_midpoints, pile_C, P_over_AEp, layer_type, Q_ult_over_AEp, tau_ult_over_AEp, S_limit_over_AEp, Q_limit_over_AEp):
    d = x[:100]
    u = x[100:]

    dz_halfel = dz/2
    strain_top = (d[:-1] - u) / dz_halfel
    strain_bottom = (u - d[1:]) / dz_halfel

    F_over_AEp_tip = jnp.minimum(Q_ult_over_AEp * Q_over_Q_ult_api(d[-1]/ pile_D[-1]), Q_limit_over_AEp)
    F_over_AEp_top_excluding_tip = strain_top
    F_over_AEp_top = jnp.append(F_over_AEp_top_excluding_tip, F_over_AEp_tip)
    
    F_over_AEp_bottom_excluding_head = strain_bottom
    F_over_AEp_bottom = jnp.insert(F_over_AEp_bottom_excluding_head, 0, P_over_AEp)

    tau_over_tau_ult = jax.lax.select(
        layer_type == 0, # 0 for clay, 1 for sand
        tau_over_tau_ult_clay_api(u / pile_D_midpoints, t_res=t_res_clay), # clay
        tau_over_tau_ult_sand_api(u / pile_D_midpoints)) # sand
    S_over_AEp = jnp.clip(pile_C * dz * tau_ult_over_AEp * tau_over_tau_ult, -S_limit_over_AEp, S_limit_over_AEp)

    zeros_1 = F_over_AEp_bottom - F_over_AEp_top
    zeros_2 = F_over_AEp_top[1:] + S_over_AEp - F_over_AEp_bottom[:-1]

    return jnp.concatenate((zeros_1, zeros_2))

def f_simultaneous_api_nondim_wrapper(x, args):
    return f_simultaneous_api_nondim(x, **args)

def solve_springs_linear_jax(pile_D, pile_L, pile_E, P, N, W):
    # W is winkler modulus, i.e. ratio between stress and *displacement* of soil
    pile_A = jnp.pi * pile_D**2 / 4
    pile_C = jnp.pi * pile_D

    z = jnp.linspace(0, 1, N)
    k_b = None # for now. this is the spring stiffness at pile base, need to check what this acc means.
    lambd = jnp.sqrt(W * pile_C / (pile_E * pile_A))
    omega = k_b / (pile_E * pile_A * lambd)
    c = (1 + omega * jnp.tanh(lambd * pile_L)) / (omega + jnp.tanh(lambd * pile_L))
    u = (P / pile_A * pile_E * lambd) * (c * jnp.cosh(lambd * z) - jnp.sinh(lambd * z))
    pass
    # note can probably just use the analytic PDE solution for this.

def solve_springs_api_jax(pile_D, pile_L, pile_E, # pile parameters
                   l_layer_type, l_gamma_d, l_e, l_c1, l_c2, l_shaft_pressure_limit, l_end_pressure_limit, l_base_depth, # soil parameters.
                   P, z_w, N=100, t_res_clay=0.9, data=None, rtol=1e-8, atol=1e-8, initial_full=0.001, throw=True, sol_verbose=frozenset()):
    """Implementation of RSPile axially loaded 1D FEM pile/soil stress/strain model using spring discretisation.
    tau_ult and q_ult calculations for sands and clays are taken from API2GEO 2011 (Reading 8.2).
    https://static.rocscience.cloud/assets/verification-and-theory/RSPile/RSPile-Axially-Loaded-Piles-Theory.pdf 

    Here I use a different set of equilibrium equations (more general) compared to solve_springs3, so that:
    - I can have an arbitrary stress-strain relationship for the concrete elements (used to be uniformly elastic)
    - I can have an arbitrary pile cross-sectional alrea profile with depth (used to be constant)

    Args:
        pile_D (pytensor array): pile diameter profile with depth
        layer_type (pytensor array): layer type profile with depth. 0 for clay, 1 for sand
        c1 (pytensor array): for each layer, the end bearing capacity factor: either N_c (clay) or N_q (sand)
        c2 (pytensor array): for each layer, either beta (sand) or psi (clay)
    """

    # Geometric constants
    pile_D_midpoints = (pile_D[1:] + pile_D[:-1]) / 2 # size N=99 to match shears
    pile_A = jnp.pi * pile_D**2 / 4 # size N=100 to match forces
    pile_A_midpoints = (pile_A[1:] + pile_A[:-1]) / 2
    pile_C = jnp.pi * pile_D_midpoints

    # Create coordinate system
    z = jnp.linspace(0, pile_L, N)
    z_midpoints = 0.5 * (z[:-1] + z[1:]) # coordinates of soil element nodes
    dz = pile_L / (N-1) # length of one element
    dz_halfel = 0.5 * dz # half the length of one element

    # Generate soil property profiles with depth (each layer is uniform)
    idxs = jnp.round(jnp.array((N-1) * l_base_depth / pile_L)).astype(int)
    idxs = jnp.diff(idxs, prepend=0)
    gamma_d = jnp.repeat(l_gamma_d, idxs, total_repeat_length=N-1)
    e = jnp.repeat(l_e, idxs, total_repeat_length=N-1)
    c1 = jnp.repeat(l_c1, idxs, total_repeat_length=N-1)
    c2 = jnp.repeat(l_c2, idxs, total_repeat_length=N-1)
    shaft_pressure_limit = jnp.repeat(l_shaft_pressure_limit, idxs, total_repeat_length=N-1)
    layer_type = jnp.repeat(l_layer_type, idxs, total_repeat_length=N-1)
    end_pressure_limit = l_end_pressure_limit[-1] # only one value for end bearing limit
    gamma_sat = gamma_d + gamma_w * e / (1 + e)
    alpha = jax.lax.select(c2 <= 1.0, 0.5 * c2**(-0.5), 0.5 * c2**(-0.25)) # only used for clay

    # Generate effective vertical stress profile with depth
    eff_stress_increments = jax.lax.select(
        z_midpoints >= z_w,
        dz * (gamma_sat - gamma_w), # above water table
        dz * gamma_d) # below water table
    eff_stress = jnp.cumsum(eff_stress_increments)

    # Generate tau_ult profile with depth
    tau_ult = jax.lax.select(
        layer_type == 0, # 0 for clay, 1 for sand
        alpha * c2 * eff_stress, # clay
        c2 * eff_stress) # sand
    
    # Get end bearing capacity
    Q_ult = pile_A[-1] * jax.lax.select(
        layer_type[-1] == 0, # 0 for clay, 1 for sand
        c1[-1] * c2[-1] * eff_stress[-1], # clay
        c1[-1] * eff_stress[-1]) # sand
    
    # Get shear force and end bearing force limits
    Q_limit = pile_A[-1] * end_pressure_limit
    S_limit = pile_C * dz * shaft_pressure_limit
    
    # Check if ultimate soil capacity is exceeded, if so return jax.nan to mark the parameter combination as invalid
    # NOTE! clay can lose capacity with high displacement, so this check needs to be refined for clay.
    # Really I will also check if the solver fails, and ignore it if it does. I should track both types of failure and report them after inference.
    Q_cap = jnp.minimum(Q_ult, Q_limit)
    S_cap = jnp.minimum(pile_C * dz * tau_ult, S_limit)
    P_cap = Q_cap + S_cap.sum()

    # initial guesses
    d_initial = jnp.full_like(z, initial_full) # d for a element N is the displacement at the bottom of the element, i.e. the pile head displacement is not included
    u_initial = jnp.full_like(z_midpoints, initial_full) # u is displacement of an element midpoints, d is displacement at nodes (i.e. element edges)
    initial = jnp.concatenate((d_initial, u_initial))
    
    args = {
        "dz": dz,"N": N, "t_res_clay": t_res_clay, "pile_D": pile_D, "pile_D_midpoints": pile_D_midpoints, "pile_C": pile_C,
        "P_over_AEp": P / (pile_A[0] * pile_E), "layer_type": layer_type, "Q_ult_over_AEp": Q_ult / (pile_A[-1] * pile_E), "tau_ult_over_AEp": tau_ult / (pile_A_midpoints * pile_E),
        "S_limit_over_AEp": S_limit / (pile_A_midpoints * pile_E), "Q_limit_over_AEp": end_pressure_limit / pile_E
    }

    # solve using optimistix
    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol, verbose=sol_verbose)

    # If ultimate capacity exceeded, return jnp.nan, otherwise solve
    params = jax.lax.cond(
        P <= P_cap,
        lambda: optx.root_find(f_simultaneous_api_nondim_wrapper, solver, initial, args=args, throw=throw, max_steps=10*N).value,
        lambda: jnp.full_like(initial, jnp.nan)
    )

    # For convergence checking
    zeros = f_simultaneous_api_nondim(params, **args)

    d = params[:N]
    u = params[N:]

    strain_top = (d[:-1] - u) / dz_halfel
    strain_tip = (u[-1] - d[-1]) / dz_halfel
    strain = jnp.append(strain_top, strain_tip)

    F = pile_A * pile_E * strain
    tau_over_tau_ult = jax.lax.select(
        layer_type == 0, # 0 for clay, 1 for sand
        tau_over_tau_ult_clay_api(u / pile_D_midpoints), # clay
        tau_over_tau_ult_sand_api(u / pile_D_midpoints)) # sand
    tau = jnp.clip(tau_ult * tau_over_tau_ult, -shaft_pressure_limit, shaft_pressure_limit)
    Q = jnp.minimum(Q_ult * Q_over_Q_ult_api(d[-1] / pile_D[-1]), pile_A[-1] * end_pressure_limit)

    return F, strain, d, u, zeros, tau, Q, eff_stress, P, P_cap
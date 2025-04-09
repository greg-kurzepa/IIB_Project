import jax
import jax.numpy as jnp
import jaxopt
import optimistix as optx
import optax

import json

m_to_in = 39.3701
gamma_w = 9.81e3 # unit weight of water

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

def f_simultaneous_api(x, l2reg, dz, N, t_res_clay, pile_D, pile_D_midpoints, pile_C, pile_E, A, P, layer_type, Q_ult, tau_ult, tau_limits, Q_limit):
    # CURRENTLY assumes pile is fully linear elastic in both directions.

    d = x[:100]
    u = x[100:]

    dz_halfel = dz/2
    strain_top = (d[:-1] - u) / dz_halfel
    strain_bottom = (u - d[1:]) / dz_halfel

    F_tip = jnp.minimum(Q_ult * Q_over_Q_ult_api(d[-1]/ pile_D[-1]), Q_limit)
    F_top_excluding_tip = A[:-1] * pile_E * strain_top
    F_top = jnp.append(F_top_excluding_tip, F_tip)
    
    F_bottom_excluding_head = A[1:] * pile_E * strain_bottom
    F_bottom = jnp.insert(F_bottom_excluding_head, 0, P)

    tau_over_tau_ult = jax.lax.select(
        layer_type == 0, # 0 for clay, 1 for sand
        tau_over_tau_ult_clay_api(u / pile_D_midpoints, t_res=t_res_clay), # clay
        tau_over_tau_ult_sand_api(u / pile_D_midpoints)) # sand
    S = pile_C * dz * jnp.clip(tau_ult * tau_over_tau_ult, -tau_limits, tau_limits)

    zeros_1 = F_bottom - F_top
    zeros_2 = F_top[1:] + S - F_bottom[:-1]

    return jnp.concatenate((zeros_1, zeros_2))

def f_simultaneous_api_nondim(x, l2reg, dz, N, t_res_clay, pile_D, pile_D_midpoints, pile_C, P_over_AEp, layer_type, Q_ult_over_AEp, tau_ult_over_AEp, tau_limits_over_AEp, Q_limit_over_AEp):
    # vals = {
    #     "dz": dz, "N": N, "t_res_clay": t_res_clay, "pile_D": pile_D, "pile_D_midpoints": pile_D_midpoints, "pile_C": pile_C,
    #     "P_over_AEp": P_over_AEp, "layer_type": layer_type, "Q_ult_over_AEp": Q_ult_over_AEp, "tau_ult_over_AEp": tau_ult_over_AEp,
    #     "tau_limits_over_AEp": tau_limits_over_AEp, "Q_limit_over_AEp": Q_limit_over_AEp
    # }
    # for key, val in vals.items():
    #     if isinstance(val, jax.Array):
    #         vals[key] = val.tolist()
    # with open('result_nondim.json', 'w') as fp:
    #     json.dump(vals, fp)
    # input("stop the code now, json nondim saved")
    
    # CURRENTLY assumes pile is fully linear elastic in both directions.

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
    S_over_AEp = pile_C * dz * jnp.clip(tau_ult_over_AEp * tau_over_tau_ult, -tau_limits_over_AEp, tau_limits_over_AEp)

    zeros_1 = F_over_AEp_bottom - F_over_AEp_top
    zeros_2 = F_over_AEp_top[1:] + S_over_AEp - F_over_AEp_bottom[:-1]

    return jnp.concatenate((zeros_1, zeros_2))

def f_simultaneous_api_nondim_wrapper(x, args):
    return f_simultaneous_api_nondim(x, None, **args)

def solve_springs_linear_jax():
    pass
    # note can probably just use the analytic PDE solution for this.

def solve_springs_api_jax(pile_D, pile_L, pile_E, # pile parameters
                   l_layer_type, l_gamma_d, l_e, l_c1, l_c2, l_shaft_friction_limit, l_end_bearing_limit, l_base_depth, # soil parameters.
                   P, z_w, N=100, t_res_clay=0.9, nondim=True, tol=1e-8, nondim_tol=1e-8, data=None):
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
    shaft_friction_limit = jnp.repeat(l_shaft_friction_limit, idxs, total_repeat_length=N-1)
    layer_type = jnp.repeat(l_layer_type, idxs, total_repeat_length=N-1)
    end_bearing_limit = l_end_bearing_limit[-1] # only one value for end bearing limit
    gamma_sat = gamma_d + gamma_w * e / (1 + e)
    alpha = jnp.select(c2 <= 1.0, 0.5 * c2**(-0.5), 0.5 * c2**(-0.25)) # only for clay

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

    # initial guesses
    # u is displacement of an element midpoints, d is displacement at nodes (i.e. element edges)
    d_initial = jnp.zeros_like(z)
    u_initial = jnp.zeros_like(z_midpoints) # d for a element N is the displacement at the bottom of the element, i.e. the pile head displacement is not included
    initial = jnp.concatenate((d_initial, u_initial))

    # CHECK if jit=true in solver will make things faster!
    
    args = {
            "dz": dz,"N": N, "t_res_clay": t_res_clay, "pile_D": pile_D, "pile_D_midpoints": pile_D_midpoints, "pile_C": pile_C,
            "pile_E": pile_E, "A": pile_A, "P": P, "layer_type": layer_type, "Q_ult": Q_ult, "tau_ult": tau_ult,
            "tau_limits": shaft_friction_limit, "Q_limit": end_bearing_limit}
    nondim_args = {
            "dz": dz,"N": N, "t_res_clay": t_res_clay, "pile_D": pile_D, "pile_D_midpoints": pile_D_midpoints, "pile_C": pile_C,
            "P_over_AEp": P / (pile_A[0] * pile_E), "layer_type": layer_type, "Q_ult_over_AEp": Q_ult / (pile_A[-1] * pile_E), "tau_ult_over_AEp": tau_ult / (pile_A_midpoints * pile_E),
            "tau_limits_over_AEp": shaft_friction_limit / (pile_A_midpoints * pile_E), "Q_limit_over_AEp": end_bearing_limit / (pile_A[-1] * pile_E)
        }

    zeros_numpy = jnp.full_like(initial, jnp.nan)
    if nondim:
        # solver = jaxopt.Broyden(fun=f_simultaneous_api, maxiter=500, implicit_diff=True, stop_if_linesearch_fails=False)
        # solver = jaxopt.ScipyRootFinding(optimality_fun=f_simultaneous_api_nondim, method="hybr", jit=False, tol=tol, use_jacrev=True)
        # print(solver.implicit_diff_solve)
        # params, state = solver.run(init_params=initial, l2reg=None, **nondim_args)

        # solve using optimistix
        solver = optx.Dogleg(rtol=0, atol=tol, verbose=frozenset({"step", "accepted", "loss", "step_size"}))
        sol = optx.root_find(f_simultaneous_api_nondim_wrapper, solver, initial, args=nondim_args, throw=False, max_steps=100*100)
        params = sol.value

        zeros = f_simultaneous_api_nondim(params, l2reg=None, **nondim_args)
        zeros_other = f_simultaneous_api(params, l2reg=None, **args)
        if data is not None:
            zeros_numpy = f_simultaneous_api_nondim(jnp.array(data), l2reg=None, **nondim_args)
    else:
        # scipy root finding worked, Broyden did not.
        solver = jaxopt.ScipyRootFinding(optimality_fun=f_simultaneous_api, method="hybr", jit=False, tol=tol, use_jacrev=True)
        params, state = solver.run(init_params=initial, l2reg=None, **args)
        zeros = f_simultaneous_api(params, l2reg=None, **args)
        zeros_other = f_simultaneous_api_nondim(params, l2reg=None, **nondim_args)

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
    tau = jnp.clip(tau_ult * tau_over_tau_ult, -shaft_friction_limit, shaft_friction_limit)
    Q = jnp.minimum(Q_ult * Q_over_Q_ult_api(d[-1] / pile_D[-1]), end_bearing_limit)

    return F, strain, d, u, zeros, tau, Q, eff_stress, zeros_other, zeros_numpy
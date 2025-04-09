#%%
import numpy as np
import jax
import jax.numpy as jnp
import json
import sys

jax.config.update('jax_enable_x64', True)

sys.path.insert(1, r"C:\Users\gregk\Documents\MyDocuments\IIB\Project\Code\Second Model\packaged")

m_to_in = 39.3701
gamma_w = 9.81e3 # unit weight of water

import _pile_and_soil

def tau_over_tau_ult_clay_api_jax(disp_over_D, t_res=0.9):
    # values to interpolate from
    z_over_D = jnp.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, jnp.inf]) / m_to_in
    tau_over_tau_ult = jnp.array([0, 0.3, 0.5, 0.75, 0.9, 1, t_res, t_res])

    # below works for positive AND negative displacement.
    return jnp.sign(disp_over_D)*jnp.interp(jnp.abs(disp_over_D), z_over_D, tau_over_tau_ult)

def tau_over_tau_ult_sand_api_jax(disp_over_D):
    # values to interpolate from
    z_over_D = jnp.array([0, 0.0016, 0.0031, 0.0057, 0.0080, 0.0100, 0.0200, jnp.inf]) / m_to_in
    tau_over_tau_ult = jnp.array([0, 0.3, 0.5, 0.75, 0.9, 1, 1, 1])

    # below works for positive AND negative displacement.
    return jnp.sign(disp_over_D)*jnp.interp(jnp.abs(disp_over_D), z_over_D, tau_over_tau_ult)

def Q_over_Q_ult_api_jax(disp_over_D):
    # values to interpolate from
    z_over_D = jnp.array([-jnp.inf, 0, 0.002, 0.013, 0.042, 0.073, 0.1, jnp.inf]) / m_to_in
    Q_over_Q_ult = jnp.array([0, 0, 0.25, 0.5, 0.75, 0.9, 1, 1])

    # below works for only downwards (+ve) displacement since going up there is no resistance.
    return jnp.interp(disp_over_D, z_over_D, Q_over_Q_ult)

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

def f_simultaneous_api_nondim_jax(x, l2reg, dz, N, t_res_clay, pile_D, pile_D_midpoints, pile_C, P_over_AEp, layer_type, Q_ult_over_AEp, tau_ult_over_AEp, tau_limits_over_AEp, Q_limit_over_AEp):
    d = x[:100]
    u = x[100:]

    dz_halfel = dz/2
    strain_top = (d[:-1] - u) / dz_halfel
    strain_bottom = (u - d[1:]) / dz_halfel

    F_over_AEp_tip = jnp.minimum(Q_ult_over_AEp * Q_over_Q_ult_api_jax(d[-1]/ pile_D[-1]), Q_limit_over_AEp)
    F_over_AEp_top_excluding_tip = strain_top
    F_over_AEp_top = jnp.append(F_over_AEp_top_excluding_tip, F_over_AEp_tip)
    
    F_over_AEp_bottom_excluding_head = strain_bottom
    F_over_AEp_bottom = jnp.insert(F_over_AEp_bottom_excluding_head, 0, P_over_AEp)

    tau_over_tau_ult = jax.lax.select(
        layer_type == 0, # 0 for clay, 1 for sand
        tau_over_tau_ult_clay_api_jax(u / pile_D_midpoints, t_res=t_res_clay), # clay
        tau_over_tau_ult_sand_api_jax(u / pile_D_midpoints)) # sand
    S_over_AEp = pile_C * dz * jnp.clip(tau_ult_over_AEp * tau_over_tau_ult, -tau_limits_over_AEp, tau_limits_over_AEp)

    zeros_1 = F_over_AEp_bottom - F_over_AEp_top
    zeros_2 = F_over_AEp_top[1:] + S_over_AEp - F_over_AEp_bottom[:-1]

    return jnp.concatenate((zeros_1, zeros_2))

def f_simultaneous_nondim(x, dz, N, pile, P_over_AEp, Q_ult_over_AEp, tau_ult_over_AEp, Q_over_Q_ult_func, tau_over_tau_ult_func, tau_limit_over_AEp, Q_limit_over_AEp):
        
        # vals = {
        #     "dz": dz, "N": N, "pile_D": pile.D, "pile_C": pile.C,
        #     "P_over_AEp": P_over_AEp, "Q_ult_over_AEp": Q_ult_over_AEp, "tau_ult_over_AEp": tau_ult_over_AEp,
        #     "tau_limits_over_AEp": tau_limit_over_AEp, "Q_limit_over_AEp": Q_limit_over_AEp
        # }
        # for key, val in vals.items():
        #     if isinstance(val, np.ndarray):
        #         vals[key] = val.tolist()
        # with open('nondim.json', 'w') as fp:
        #     json.dump(vals, fp)
        # input("stop the code now, json saved")

        d = x[:N]
        u = x[N:]

        dz_halfel = dz/2
        strain_top = (d[:-1] - u) / dz_halfel
        strain_bottom = (u - d[1:]) / dz_halfel

        F_over_AEp_tip = min(Q_ult_over_AEp * Q_over_Q_ult_func(d[-1]), Q_limit_over_AEp)
        F_over_AEp_top_excluding_tip = strain_top
        F_over_AEp_top = np.append(F_over_AEp_top_excluding_tip, F_over_AEp_tip)
        
        F_over_AEp_bottom_excluding_head = strain_bottom
        F_over_AEp_bottom = np.insert(F_over_AEp_bottom_excluding_head, 0, P_over_AEp)

        S_over_AEp = pile.C * dz * np.clip(tau_ult_over_AEp * tau_over_tau_ult_func(u), -tau_limit_over_AEp, tau_limit_over_AEp)

        zeros_1 = F_over_AEp_bottom - F_over_AEp_top
        zeros_2 = F_over_AEp_top[1:] + S_over_AEp - F_over_AEp_bottom[:-1]

        return np.concatenate((zeros_1, zeros_2))

# I will load in the json file I made earlier, and pass the arguments into both functions, and see if they give the same output.
with open(r"C:\Users\gregk\Documents\MyDocuments\IIB\Project\Code\Second Model\result_nondim_jax.json") as json_data:
    data = json.load(json_data)
with open(r"C:\Users\gregk\Documents\MyDocuments\IIB\Project\Code\Second Model\disp_nondim.json") as json_data:
    disp_nondim = np.array(json.load(json_data))
with open(r"C:\Users\gregk\Documents\MyDocuments\IIB\Project\Code\Second Model\disp_nondim_jax.json") as json_data:
    disp_nondim_jax = np.array(json.load(json_data))

dz = data["dz"]
N = data["N"]
t_res_clay = data["t_res_clay"]
pile_D = np.array(data["pile_D"])
pile_D_midpoints = np.array(data["pile_D_midpoints"])
pile_C = np.array(data["pile_C"])
P_over_AEp = data["P_over_AEp"]
layer_type = np.array(data["layer_type"])
Q_ult_over_AEp = data["Q_ult_over_AEp"]
tau_ult_over_AEp = np.array(data["tau_ult_over_AEp"])
tau_limits_over_AEp = np.array(data["tau_limits_over_AEp"])
Q_limit_over_AEp = data["Q_limit_over_AEp"]

A = np.full_like(pile_D, np.pi * 0.3**2)
A_midpoints = (A[1:] + A[:-1]) / 2

pile = _pile_and_soil.Pile(R=0.3, L=30, E=35e9)

tau_over_tau_ult_func = lambda u : np.where(layer_type == 0, tau_over_tau_ult_clay(u / pile.D, t_res=t_res_clay), tau_over_tau_ult_sand(u / pile.D))
Q_over_Q_ult_func = lambda d_tip : Q_over_Q_ult(d_tip / pile.D)

args = {
    "dz": dz, "N": N, "pile": pile,
    "P_over_AEp": P_over_AEp, "Q_ult_over_AEp": Q_ult_over_AEp, "tau_ult_over_AEp": tau_ult_over_AEp,
    "Q_over_Q_ult_func": Q_over_Q_ult_func, "tau_over_tau_ult_func": tau_over_tau_ult_func,
    "tau_limit_over_AEp": tau_limits_over_AEp, "Q_limit_over_AEp": Q_limit_over_AEp
}

args_jax = {
    "l2reg": None, "dz": dz, "N": N, "t_res_clay": t_res_clay, "pile_D": jnp.array(pile_D),
    "pile_D_midpoints": jnp.array(pile_D_midpoints), "pile_C": jnp.array(pile_C),
    "P_over_AEp": P_over_AEp, "layer_type": jnp.array(layer_type), "Q_ult_over_AEp": Q_ult_over_AEp,
    "tau_ult_over_AEp": jnp.array(tau_ult_over_AEp), "tau_limits_over_AEp": jnp.array(tau_limits_over_AEp),
    "Q_limit_over_AEp": Q_limit_over_AEp
}

x = np.zeros(2*N-1)
# x = disp_nondim
res = f_simultaneous_nondim(x, **args)
res_jax = f_simultaneous_api_nondim_jax(x, **args_jax)
diff = res - res_jax
print("Difference between JAX and non-JAX: ", diff)
# %%

#%%
import jax
import jax.numpy as jnp
import jaxopt

# EITHER use smooth approximation
# OR use dedicated root-finding tools (Broyden)
# Broyden is more accurate!

def smooth_abs(x, epsilon=1e-6):
    return jnp.sqrt(x**2 + epsilon)  # Smooth approximation to abs(x)

def minimise_f(params, l2reg, roots):
    f = params - roots
    return jnp.sum(smooth_abs(f))
def minimise_f_root(params, l2reg, roots):
    return params - roots

def solve_system(roots):
    init_params = jnp.array([0.0, 0.0])
    # solver = jaxopt.LBFGS(fun=minimise_f, maxiter=500, implicit_diff=True)
    solver = jaxopt.Broyden(fun=minimise_f_root, maxiter=500, implicit_diff=True)
    params, state = solver.run(init_params=init_params, l2reg=None, roots=roots)

    return 1*params[0] + 5*params[1]

roots = jnp.array([10.0, 22.0])
gv1 = jax.value_and_grad(solve_system)
print(gv1(roots))
# %%

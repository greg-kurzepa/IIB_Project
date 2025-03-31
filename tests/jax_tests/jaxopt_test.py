#%%
import jax
import jax.numpy as jnp
import jaxopt

def minimise_f(params, l2reg, roots):
    # l2reg is not used here. We simply pick params such that the output is minimised.
    # I actually want to find the function roots. Thus the output will have abs().

    # Define a function with 2 inputs and 2 outputs, and root is params = (1,2)
    f = params - roots
    # Multiply the function by some numbers that will not change the root but provide
    # an example of being able to manipulate the output
    f *= jnp.array([3.0, 2.5])

    return f

def solve_system(roots):
    init_params = jnp.array([0.0, 0.0])
    # Broyden is the built-in jaxopt solver for root finding. It is a quasi-Newton method.
    solver = jaxopt.Broyden(fun=minimise_f, maxiter=500, implicit_diff=True, stop_if_linesearch_fails=True)
    params, state = solver.run(init_params=init_params, l2reg=None, roots=roots)

    # return params
    return 1*params[0] + 5*params[1] # for grad testing purposes, we return the sum of the params

def solve_system_mirror(roots):
    return 1*roots[0] + 5*roots[1]

#%%

# test gradients.
# gradient wrt root 1 should be 1, and wrt root 2 should be 5

roots = jnp.array([10.0, 22.0])
gv1 = jax.value_and_grad(solve_system)
print(gv1(roots))
gv2 = jax.value_and_grad(solve_system_mirror)
print(gv2(roots))
# %%

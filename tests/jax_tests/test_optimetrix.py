#%%
import jax
import jax.numpy as jnp
import optimistix as optx

# Often import when doing scientific work
jax.config.update("jax_enable_x64", True)

def fn_wrapper(y, args):
    return fn(y, **args)

def fn(y, roots):
    return y - roots

def solve(roots):
    # solver = optx.Newton(rtol=1e-8, atol=1e-8)
    solver = optx.Dogleg(rtol=1e-8, atol=1e-8)
    y0 = jnp.array([0,0])
    sol = optx.root_find(fn_wrapper, solver, y0, args={"roots":roots})
    return sol.value[0] + 5*sol.value[1]

# %%

vg = jax.value_and_grad(solve, argnums=0)
roots = jnp.array([1.0, 2.0])
result = vg(roots)
print(result)

#%%
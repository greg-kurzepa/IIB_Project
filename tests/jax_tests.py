#%%
import jax
import jax.numpy as jnp
import jax.scipy.optimize as opt

def out_fun(minloc):
    def f(x): return ((x-minloc)**2 + 2)[0] # minimum at x=2
    # def g(x): return (x**2 + 2)[0] # minimum at x=0

    # def fun(x_guess):
    #     # the minimize should always return 0
    #     placeholder0 = opt.minimize(g, jnp.array([0.0]), method="BFGS").fun
    #     return f(x_guess) + placeholder0

    bases = jnp.array([1.0,2.0,3.0])

    min_f = opt.minimize(f, jnp.array([0.0]), method="BFGS").x[0]
    return bases * min_f

def wrap_fun(x):
    return jnp.sum(out_fun(x))

print(out_fun(2.0))
grad = jax.grad(wrap_fun)
# %%

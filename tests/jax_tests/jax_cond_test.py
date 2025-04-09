#%%
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

a = jnp.array([1, 2, 3], dtype=jnp.float64)
b = jnp.where(3 <= 2, a, jnp.nan * a)
c = jax.lax.cond(3 <= 2, lambda: a, lambda: jnp.nan * a)

print(b, c)
# %%

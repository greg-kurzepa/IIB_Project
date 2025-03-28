#%%
import numpy as np
import scipy

P = 1.8e6
A = np.pi*0.3**2
C = np.pi*0.6
E_p = 35e9
E_s = E_p/100
dz = 5
dm = dz/2
Q = 0

b = np.array([P, 0, Q, P, -Q])

a = np.array([
    [A*E_p/dm, 0, 0, -A*E_p/dm, 0],
    [0, -2, 0, 1, 1],
    [0, 0, -A*E_p/dm, 0, A*E_p/dm],
    [0, A*E_p/dm - C*dz*E_s/dm, 0, C*dz*E_s/dm, -A*E_p/dm],
    [0, A*E_p/dm, -C*dz*E_s/dm, -A*E_p/dm, C*dz*E_s/dm]
])

print(a)
print(np.linalg.cond(a))
try:
    inv = np.linalg.inv(a)
    print(f"Invertible: solution is {inv @ b} or {np.linalg.solve(a, b)}")
except np.linalg.LinAlgError:
    print("Not invertible")
# %%

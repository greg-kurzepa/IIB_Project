import numpy as np
import pytensor
import pytensor.tensor as pt

x = pt.vector(name="x")

y = pt.linspace(x[0], x[1], x[2])

a = np.array([0, 5, 10, 20])
b = np.array([0, 5, 20, 40])

pytensor.dprint(z)

f = pytensor.function(inputs=[x], outputs=[z])
print(f(x=[0, 10, 6]))
#%%
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, x0=0, k=1, A=1):
    """
    Sigmoid function.

    Args:
        x (float): Input value.
        x0 (float): x-value of the sigmoid's midpoint.
        k (float): steepness of the curve.
        A (float): maximum value of the curve.

    Returns:
        float: Output value of the sigmoid function.
    """
    return A / (1 + np.exp(-k * (x - x0)))

def f1(x):
    return 0.5*x
def f2(x):
    return x
def smoothed(x, k):
    s = sigmoid(x, k=k)
    return (1-s)*f1(x) + s*f2(x)

x = np.linspace(-10, 10, 10000)
k = 10

piecewise = np.where(
    x < 0,
    f1(x),
    f2(x)
)

plt.plot(x, piecewise, label='Piecewise')
plt.plot(x, smoothed(x, k), label=f'Smoothed (k={k})')
plt.grid()
plt.legend()
plt.show()
# %%

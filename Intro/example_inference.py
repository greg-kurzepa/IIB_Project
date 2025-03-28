#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime

import pymc as pm

def f(m, c, x):
    # straight line function
    return m * x + c

def log_likelihood_f(m, c, x, y, sigma):
    # m, c are the thetas
    # x is the coordinate system
    # y is the data
    # sigma (as well as m, c) is needed to construct the likelihood pdf

    logp = - np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - f(m, c, x)) / sigma) ** 2
    return logp.sum()

class LogLikelihood(Op):
    def make_node(self, m, c, x, y, sigma) -> Apply:
        # Convert inputs to tensor variables
        m = pt.as_tensor(m)
        c = pt.as_tensor(c)
        x = pt.as_tensor(x)
        y = pt.as_tensor(y)
        sigma = pt.as_tensor(sigma)

        inputs = [m, c, x, y, sigma]
        # Define output type, in our case a single likelihood value
        outputs = [pt.dscalar()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)
    
    def perform(self, node : Apply, inputs : list[np.ndarray], outputs : list[list[None]]):
        # the inputs are a list of ndarrays - each item in the list is one of m, c, x, y, sigma
        m, c, x, y, sigma = inputs
        log_likelihood = log_likelihood_f(m, c, x, y, sigma)
        outputs[0][0] = np.asarray(log_likelihood)

# set up our data
N = 10  # number of data points
sigma = 1.0  # standard deviation of noise
x = np.linspace(0.0, 9.0, N)

mtrue = 0.4  # true gradient
ctrue = 3.0  # true y-intercept

truemodel = f(mtrue, ctrue, x)

# make data
rng = np.random.default_rng(716743)
data = sigma * rng.normal(size=N) + truemodel

# create our Op
loglike_op = LogLikelihood()
# create a wrapper function
def custom_dist_loglike(data, m, c, x, sigma):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(m, c, x, data, sigma)

test_out = loglike_op(mtrue, ctrue, x, data, sigma)
# %%

# use PyMC to sampler from log-likelihood
with pm.Model() as no_grad_model:
    # uniform priors on m and c
    m = pm.Uniform("m", lower=-10.0, upper=10.0, initval=mtrue)
    c = pm.Uniform("c", lower=-10.0, upper=10.0, initval=ctrue)

    # use a CustomDist with a custom logp function
    likelihood = pm.CustomDist(
        "likelihood", m, c, x, sigma, observed=data, logp=custom_dist_loglike
    )

print("Sampling...")
with no_grad_model:
    # Use custom number of draws to replace the HMC based defaults
    idata_no_grad = pm.sample(3000, tune=1000)
print("Done sampling!")

# plot the traces
az.plot_trace(idata_no_grad, lines=[("m", {}, mtrue), ("c", {}, ctrue)]);
# %%

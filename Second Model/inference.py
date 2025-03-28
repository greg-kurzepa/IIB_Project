#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

print(f"Running on PyMC v{pm.__version__}")

az.style.use("arviz-darkgrid")

#%% --------------------------------------------------------------------------------------------
# Step 0: Define the likelihood functions and the black box Op

def forward_model(pile, soil, P, z_w, N=100, t_res_clay=0.9):
    res = _model_springs.solve_springs4(pile, soil, P, z_w, N=N, t_res_clay=t_res_clay, tau_over_tau_ult_func = None, Q_over_Q_ult_func = None, tol=1e-8, outtol=1e-3)
    return res[0]

def log_likelihood_f(sigma, data, *forward_params):
    # assuming additive gaussian white noise
    logp = - np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((data - forward_model(*forward_params)) / sigma) ** 2
    return logp.sum()

# define a pytensor Op for our likelihood function
class LogLikelihood(Op):
    def make_node(self, data, sigma, gamma_d) -> Apply:
        # Convert inputs to tensor variables
        data = pt.as_tensor(data)
        sigma = pt.as_tensor(sigma)
        gamma_d = pt.as_tensor(gamma_d)

        inputs = [data, sigma, gamma_d]
        # Define output type, in this case a single scalar
        outputs = [pt.scalar()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output given numerical inputs.
        data, sigma, gamma_d = inputs

        soil = _pile_and_soil.Soil()
        for layer_i in range(gamma_d.shape[0]):
            soil.add_layer(_pile_and_soil.SandLayer(gamma_d=gamma_d[layer_i], e=0.689, N_q=8, beta=plug_factor*0.214, shaft_friction_limit=47.8e3, end_bearing_limit=1.9e6, base_depth=layer_base_depths[layer_i]))

        # call our numpy log-likelihood function
        log_likelihood = log_likelihood_f(sigma, data, *(pile, soil, P, z_w, N))

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(log_likelihood)

# create our Op
loglike_op = LogLikelihood()

#%% --------------------------------------------------------------------------------------------
# Step 1: Define the system and ground truth parameters. Define the synthetic data.

# Pile
pile = _pile_and_soil.Pile(R=0.3, L=30, E=35e9)

# Define the other variables (loading, number of elements, model noise variance)
P = 1.8e6 # top axial load
N = 100 # number of nodes along pile
z = np.linspace(0, pile.L, N)
z_midpoints = 0.5 * (z[:-1] + z[1:])

plug_factor = 1.25 # according to API, beta for soils should be scaled by 1.25 for plugged piles
layer_base_depths = [12.5, pile.L]
layer1 = _pile_and_soil.SandLayer(gamma_d=15e3, e=0.689, N_q=8, beta=plug_factor*0.214, shaft_friction_limit=47.8e3, end_bearing_limit=1.9e6, base_depth=layer_base_depths[0])
layer2 = _pile_and_soil.SandLayer(gamma_d=17e3, e=0.441, N_q=40, beta=plug_factor*0.46, shaft_friction_limit=96e3, end_bearing_limit=10e6, base_depth=layer_base_depths[1])
soil = _pile_and_soil.Soil([layer1, layer2])

# these parameters are known to the model
sigma = 0.1e6 # standard deviation of noise
z_w = 3 # water table depth

truemodel = forward_model(pile, soil, P, z_w, N)

# make data
rng = np.random.default_rng(716743)
data = sigma * rng.normal(size=N) + truemodel

#%% --------------------------------------------------------------------------------------------
# Step 2: Define the inference model and the priors on the parameters.
# A sensitivity analysis determined gamma_d, beta and shaft_friction limit were the most important parameters. I will additionally infer N_q which is coupled to shaft_friction_limit in a way.
# But to start with just gamma_d.
# Assume we know the rough category of soil e.g. loose or dense sand

def wrapped_log_likelihood(data, sigma, gamma_d):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(data, sigma, gamma_d)

# Priors for each layer
gamma_d_mu = [np.log(15e3), np.log(18.5e3)]
gamma_d_sigma = [1.5e3, 1.5e3]

# use PyMC to sampler from log-likelihood
with pm.Model() as no_grad_model:

    gamma_d = pm.LogNormal("gamma_d", mu=gamma_d_mu, sigma=gamma_d_sigma, shape=2)

    # use a CustomDist with a custom logp function
    likelihood = pm.CustomDist(
        "likelihood", sigma, gamma_d, observed=data, logp=wrapped_log_likelihood
    )

#%%

with no_grad_model:
    # Use custom number of draws to replace the HMC based defaults
    idata_no_grad = pm.sample(3000, tune=1000, cores=1)

#%%

# plot the traces
# az.plot_trace(idata_no_grad, lines=[("m", {}, mtrue), ("c", {}, ctrue)])
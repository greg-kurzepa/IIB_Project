#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import pytensor.gradient
import copy
import multiprocessing

import jax
import jax.numpy as jnp
import jax.scipy.optimize as opt

# Here I demonstrate the various jax features I want to use in the pymc model.
# and that they work with grad and pymc to allow me to use HMC sampling.

# The soil 'unit weights' of two layers are specified, and these will be inferred by pymc.
# The data is generated synthetically from the two true unit weights.

if __name__ == "__main__":

    def f1(x):
        return 2*x

    def f2(x):
        return 3*x

    def forward(unit_weights, layer_depths, N, L, threshold=2.5):
        # generate scaled unit weight profile
        z = jnp.linspace(0, L, N)
        idxs = jnp.round(jnp.array(N * layer_depths / L)).astype(int)
        idxs = jnp.diff(idxs, prepend=0)
        profile = jnp.repeat(unit_weights, idxs, total_repeat_length=N)
        scaled_profile = jax.lax.select(profile < threshold, f1(profile), f2(profile))

        return scaled_profile

    def forward_log_likelihood(unit_weights, layer_depths, data, sigma, N, L, threshold=2.5):
        scaled_profile = forward(unit_weights, layer_depths, N, L, threshold)
        
        # find likelihood of data given scaled profile and white gaussian noise
        logp = - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((data - scaled_profile) / sigma) ** 2
        return logp.sum()

    # using JIT has a MASSIVE impact! Made inference go from a few minutes to a few SECONDS.
    jitted_logp = jax.jit(forward_log_likelihood, static_argnames=("N", "L", "threshold"))
    jitted_logp_grad = jax.jit(jax.grad(forward_log_likelihood, argnums=[0]), static_argnames=("N", "L", "threshold"))

    # define a pytensor Op for our likelihood function
    class LogLikelihood(Op):
        def make_node(self, data, sigma, unit_weights, layer_depths) -> Apply:
            # Convert inputs to tensor variables
            # Note the difference with black box op: pt.as_tensor_variable instead of pt.as_tensor
            inputs = [
                pt.as_tensor_variable(data),
                pt.as_tensor_variable(sigma),
                pt.as_tensor_variable(unit_weights),
                pt.as_tensor_variable(layer_depths),
            ]

            # Define output type, in this case a single scalar
            outputs = [pt.scalar()]

            # Apply is an object that combines inputs, outputs and an Op (self)
            return Apply(self, inputs, outputs)

        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            # This is the method that compute numerical output given numerical inputs.
            data, sigma, unit_weights, layer_depths = inputs

            # call our numpy log-likelihood function
            # also try without jit?
            log_likelihood = jitted_logp(unit_weights, layer_depths, data, sigma, N, L)

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            outputs[0][0] = np.asarray(log_likelihood, dtype=node.outputs[0].dtype)

        def grad(self, inputs, output_gradients):
            grad_wrt_unit_weights = logp_grad_op(*inputs)
            # If there are inputs for which the gradients will never be needed or cannot
            # be computed, `pytensor.gradient.grad_not_implemented` should  be used as the
            # output gradient for that input.
            output_gradient = output_gradients[0]
            return [
                pytensor.gradient.grad_not_implemented(self, 0, inputs[0]),
                pytensor.gradient.grad_not_implemented(self, 1, inputs[1]),
                output_gradient * grad_wrt_unit_weights,
                pytensor.gradient.grad_not_implemented(self, 3, inputs[3]),]

    # define a pytensor Op for the gradient of our likelihood function
    class LogLikelihoodGrad(Op):
        def make_node(self, data, sigma, unit_weights, layer_depths) -> Apply:
            # Convert inputs to tensor variables
            # Note the difference with black box op: pt.as_tensor_variable instead of pt.as_tensor
            inputs = [
                pt.as_tensor_variable(data),
                pt.as_tensor_variable(sigma),
                pt.as_tensor_variable(unit_weights),
                pt.as_tensor_variable(layer_depths),
            ]

            # In practice, you should use
            # the exact dtype to avoid overhead when saving the results of the computation
            # in `perform`
            print(inputs[2].type())
            outputs = [inputs[2].type(),]

            # Apply is an object that combines inputs, outputs and an Op (self)
            return Apply(self, inputs, outputs)

        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            data, sigma, unit_weights, layer_depths = inputs

            (grad_wrt_unit_weights,) = jitted_logp_grad(unit_weights, layer_depths, data, sigma, N, L)

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            outputs[0][0] = np.asarray(grad_wrt_unit_weights, dtype=node.outputs[0].dtype)
            # outputs[1][0] = 

    # create our Ops
    logp_op = LogLikelihood()
    logp_grad_op = LogLikelihoodGrad()

    #%%

    # generate true and observed data
    N = 100
    L = 10.0
    sigma = 0.1
    random_seed = 42
    unit_weights = np.array([2.0, 3.0])#, dtype=np.float64)
    layer_depths = np.array([5.0, L])#, dtype=np.float64)
    true_profile = np.array(forward(unit_weights, layer_depths, N, L))#, dtype=np.float64)
    data = true_profile + np.random.normal(0, sigma, N)

    # confirm everything was specified correctly
    print("forward",forward_log_likelihood(unit_weights, layer_depths, data, sigma, N, L))
    print(logp_op(data, sigma, unit_weights, layer_depths).eval())
    print(jitted_logp_grad(unit_weights, layer_depths, data, sigma, N, L)[0])
    print(logp_grad_op(data, sigma, unit_weights, layer_depths).eval())

    #%%

    # define the pymc model

    def get_lognormal_params(mean, stdev):
        # The mu, sigma parameters of a lognormal distribution are not the true mean and standard deviation of the distribution.
        # This function takes the true mean and standard deviation and outputs the mu, sigma parameters of the lognormal distribution.
        mu_lognormal = np.log(mean**2 / np.sqrt(stdev**2 + mean**2))
        sigma_lognormal = np.sqrt(np.log(1 + (stdev**2 / mean**2)))
        return {"mu" : mu_lognormal, "sigma" : sigma_lognormal}

    # Priors for each layer
    unit_weights_mean = np.array([2.3, 2])#, dtype=np.float64)
    unit_weights_stdev = np.array([1, 1])#, dtype=np.float64)

    def wrapped_logp(data, sigma, unit_weights):
        print(unit_weights)
        return logp_op(data, sigma, unit_weights, layer_depths)

    model = pm.Model()
    with model:

        # priors, i.e. p(theta)
        unit_weight = pm.LogNormal("unit_weight", shape=2, **get_lognormal_params(unit_weights_mean, unit_weights_stdev))

        # likelihood, i.e. p(x|theta)
        # in order to do prior/posterior predictive checks, it needs the 'random' argument which allows is to generate one sample from the pdf
        out = pm.CustomDist(
            "out", sigma, unit_weight, observed=data, logp=wrapped_logp
        )

    pm.model_to_graphviz(model)

    initial_point = model.initial_point()
    print(initial_point)
    print(model.point_logps(initial_point))

    #%%

    with model:
        idata = pm.sample(draws=100, tune=100, chains=1, cores=1)

    #%%

    az.plot_trace(idata, var_names=["unit_weight"])
    plt.show()
    az.plot_dist_comparison(idata, var_names=["unit_weight"])
    plt.show()
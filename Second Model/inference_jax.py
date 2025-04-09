## NOTES
## CURRENTLY NO SUPPORT FOR INFERRING SIGMA.
# I will have to think carefully about how to implement this. Sigma is only used in the likelihood function
# and not the forward function, so it shouldn't go in the forward params dicts.
# It also shouldn't always be passed into the grad op because that causes errors when it is fixed
# although might be worth trying that again, it could have been a bug in my code.

#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import copy
import jax
import jax.numpy as jnp
import equinox
import multiprocessing

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs_jax as _model_springs

jax.config.update('jax_enable_x64', True) #ESSENTIAL

if __name__ == "__main__":
    print(f"Running on PyMC v{pm.__version__}")

    az.style.use("arviz-darkgrid")

    # Utility function for plotting function pdf
    def plot_xY(x, Y, ax, label=None):
        quantiles = Y.quantile((0.025, 0.25, 0.5, 0.75, 0.975), dim=("chain", "draw")).transpose()

        az.plot_hdi(
            x,
            hdi_data=quantiles.sel(quantile=[0.025, 0.975]),
            fill_kwargs={"alpha": 0.25},
            smooth=False,
            ax=ax,
        )
        az.plot_hdi(
            x,
            hdi_data=quantiles.sel(quantile=[0.25, 0.75]),
            fill_kwargs={"alpha": 0.5},
            smooth=False,
            ax=ax,
        )
        ax.plot(x, quantiles.sel(quantile=0.5), color="C1", lw=3, label=label)

    # The mu, sigma parameters of a lognormal distribution are not the true mean and standard deviation of the distribution.
    # This function takes the true mean and standard deviation and outputs the mu, sigma parameters of the lognormal distribution.
    def get_lognormal_params(mean, stdev):
        mu_lognormal = np.log(mean**2 / np.sqrt(stdev**2 + mean**2))
        sigma_lognormal = np.sqrt(np.log(1 + (stdev**2 / mean**2)))
        return {"mu" : mu_lognormal, "sigma" : sigma_lognormal}

    #%% --------------------------------------------------------------------------------------------
    # Step 1: Define the parameters. Define which parameters are to be inferred and which are known. Define the priors.

    # These parameters are never inferred
    N=100
    fixed_forward_params_dict = {
        "P" : 1.8e6, # top axial load
        "N" : N, # number of nodes along pile
        "z_w" : 3, # water table depth
        "l_layer_type" : np.array([1, 1]), # 0 for clay, 1 for sand
        "l_base_depth" : np.array([12.5, 30]), # base depth of each layer (final value is the length of the pile)
        "pile_L" : 30, # pile length
        "pile_D" : np.full(N, 0.6), # pile diameter
        "pile_E" : 35e9, # pile elastic modulus
        "t_res_clay" : 0.9
    }

    # Sigma is only used in the log likelihood and not the forward model, so is left out of the above dictionary
    sigma = 0.05e6 # standard deviation of noise
    z = np.linspace(0, fixed_forward_params_dict["pile_L"], fixed_forward_params_dict["N"])

    # For below:
    # c1 is N_c for clay and N_q for sand
    # c2 is beta for sand and psi for clay
    # shaft_friction_limit and end_bearing_limit are only ever inferred for sand. for clay, they are always inf.
    # I will need a way to implement that in the model.

    # Contains the ground truth values of the parameters for each layer that *could* be inferred
    sand_plug_factor = 1.25 # according to API, beta for sands should be scaled by 1.25 for plugged piles
    inferrable_forward_params_dict = {
        "l_gamma_d" : np.array([15e3, 17e3]),
        "l_e" : np.array([0.689, 0.441]),
        "l_c1" : np.array([8, 40]),
        "l_c2" : np.array([sand_plug_factor*0.214, sand_plug_factor*0.46]),
        "l_shaft_pressure_limit" : np.array([47.8e3, 96e3]),
        "l_end_pressure_limit" : np.array([1.9e6, 10e6]),
    }

    # Contains the parameters to be inferred and their prior distributions
    # wrapper_fun takes mean, stdev as input and returns the parameters to input to dist
    # Prior widths I set here reflect knowledge the rough category of soil e.g. loose or dense sand
    inferred_forward_params_dict = {
        "l_gamma_d" : {
            "dist" : pm.LogNormal,
            "mean" : np.array([15e3, 17e3]),
            "stdev" : np.array([3e3, 3e3]),
            "wrapper_fun" : get_lognormal_params,
        }
    }
    not_inferred_forward_params_dict = {key : inferrable_forward_params_dict[key] for key in inferrable_forward_params_dict if key not in inferred_forward_params_dict.keys()}

    # In the actual code, these parameters will be passed around as tuples, not dicts.
    # This is because jax.grad() only accepts positional arguments and writing a wrapper that takes account of the argnums argument was painful.

    # forward_arg_order is the order to pass into the *forward* function
    forward_arg_order = ("pile_D", "pile_L", "pile_E", "l_layer_type", "l_gamma_d", "l_e", "l_c1", "l_c2", "l_shaft_pressure_limit", "l_end_pressure_limit", "l_base_depth", "P", "z_w", "N", "t_res_clay")
    static_argnames = ("pile_L", "pile_E", "P", "z_w", "N", "t_res_clay")
    forward_static_argnums = [forward_arg_order.index(x) for x in static_argnames]
    
    # likelihood_arg_order is the order to pass into the *likelihood* function
    likelihood_arg_order = ("data", "sigma") + forward_arg_order
    likelihood_static_argnums = [likelihood_arg_order.index(x) for x in static_argnames]
    grad_argnums = [likelihood_arg_order.index(key) for key in inferred_forward_params_dict.keys()]

    # wrapper_arg_order is the order to pass into the *wrapped* likelihood & grad functions, excluding data, sigma which are always first
    wrapper_arg_order = (*inferred_forward_params_dict.keys(), *fixed_forward_params_dict.keys(), *not_inferred_forward_params_dict.keys())

    # When using the Ops, only the inferred parameters are passed for simplicity, in the following order:
    # op_arg_order = ("data", "sigma", *inferred_forward_params_dict.keys())

    # When using ground truth into the forward function (skipping the likelihood function), we use inferrable parameters.
    truth_arg_order = (*inferrable_forward_params_dict.keys(), *fixed_forward_params_dict.keys())

    #%% --------------------------------------------------------------------------------------------
    # Step 2: Define the likelihood functions and the jax Ops

    def forward_model(*forward_params):
        res = _model_springs.solve_springs_api_jax(*forward_params, throw=False)
        return res[0]

    counter = 0
    def forward_log_likelihood(data, sigma, *forward_params):
        global counter
        # Assuming additive gaussian white noise
        # If results are all jnp.nan, reflecting invalid parameter combination, set probability to zero
        res = forward_model(*forward_params)
        logp = jax.lax.cond(
            jnp.all(jnp.isnan(res)),
            lambda: -jnp.inf,
            lambda: (-jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((data - res) / sigma) ** 2).sum()
        )
        concrete_logp = jax.lax.stop_gradient(logp)
        concrete_gamma_d = jax.lax.stop_gradient(forward_params[grad_argnums[0]-2])
        # print(f"counter: {counter}, logp: {jax.debug.print("Traced value: {}", concrete_value)}")
        print(f"counter: {counter}, gamma_d: {concrete_gamma_d}, logp: {concrete_logp})")
        counter+=1
        return logp
    
    jitted_forward = jax.jit(forward_model, static_argnums=forward_static_argnums)
    jitted_logp = jax.jit(forward_log_likelihood, static_argnums=likelihood_static_argnums)
    jitted_logp_grad = jax.jit(jax.grad(forward_log_likelihood, argnums=grad_argnums), static_argnums=likelihood_static_argnums)

    # jitted_forward = forward_model
    # jitted_logp = forward_log_likelihood
    # jitted_logp_grad = jax.grad(forward_log_likelihood, argnums=grad_argnums)

    def sort_x_by_y(x, y):
        return [X for (Y,X) in sorted(zip(y,x), key=lambda pair: pair[0])]

    def reorder_params(*unordered_forward_params, unordered_argnames=wrapper_arg_order, ordered_argnames=forward_arg_order):
        unordered_idx_in_ordered = [ordered_argnames.index(x) for x in unordered_argnames]
        ordered_forward_params = sort_x_by_y(unordered_forward_params, unordered_idx_in_ordered)
        return ordered_forward_params

    def jitted_logp_wrapper(data, sigma, *unordered_forward_params):
        return jitted_logp(data, sigma, *reorder_params(*unordered_forward_params))
    
    def jitted_logp_grad_wrapper(data, sigma, *unordered_forward_params):
        return jitted_logp_grad(data, sigma, *reorder_params(*unordered_forward_params))

    # define a pytensor Op for our likelihood function
    class LogLikelihood(Op):
        # The inputs are parameters to be inferred. The other parameters are used from the global values.
        def make_node(self, *drawn_inferred_forward_params) -> Apply:
            # Convert inputs to tensor variables
            # Note the difference with black box op: pt.as_tensor_variable instead of pt.as_tensor
            inputs = [pt.as_tensor_variable(x) for x in drawn_inferred_forward_params]

            # Define output type, in this case a single scalar
            outputs = [pt.scalar()]

            # Apply is an object that combines inputs, outputs and an Op (self)
            return Apply(self, inputs, outputs)

        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            # This is the method that compute numerical output given numerical inputs.

            log_likelihood = jitted_logp_wrapper(data, sigma, *inputs, *fixed_forward_params_dict.values(), *not_inferred_forward_params_dict.values())

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            outputs[0][0] = np.asarray(log_likelihood, dtype=node.outputs[0].dtype)

        def grad(self, inputs, output_gradients):
            out = logp_grad_op(*inputs)
            grads = out if isinstance(out, tuple) else (out,)

            # If there are inputs for which the gradients will never be needed or cannot
            # be computed, `pytensor.gradient.grad_not_implemented` should  be used as the
            # output gradient for that input.
            output_gradient = output_gradients[0]

            print("grad inputs: ", inputs, "grad: ", grads)
            return [output_gradient * grads[i] for i, inp in enumerate(inputs)]

    # define a pytensor Op for the gradient of our likelihood function
    class LogLikelihoodGrad(Op):
        def make_node(self, *drawn_inferred_forward_params) -> Apply:
            # Convert inputs to tensor variables
            # Note the difference with black box op: pt.as_tensor_variable instead of pt.as_tensor
            inputs = [pt.as_tensor_variable(x) for x in drawn_inferred_forward_params]

            # In practice, you should use
            # the exact dtype to avoid overhead when saving the results of the computation
            # in `perform`
            outputs = [x.type() for x in inputs]

            # Apply is an object that combines inputs, outputs and an Op (self)
            return Apply(self, inputs, outputs)

        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            grads = jitted_logp_grad_wrapper(data, sigma, *inputs, *fixed_forward_params_dict.values(), *not_inferred_forward_params_dict.values())

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            for i, grad in enumerate(grads):
                outputs[i][0] = np.asarray(grad, dtype=node.outputs[i].dtype)

    # create our Ops
    logp_op = LogLikelihood()
    logp_grad_op = LogLikelihoodGrad()

    #%% --------------------------------------------------------------------------------------------
    # Step 3: Generate ground truth and synthetic data.

    # Generate ground truth data
    unordered_true_params = (*inferrable_forward_params_dict.values(), *fixed_forward_params_dict.values())
    ordered_true_params = reorder_params(*unordered_true_params, unordered_argnames=truth_arg_order)
    true_forces = np.array(forward_model(*ordered_true_params))

    # Generate synthetic observed data
    random_seed = 716743
    rng = np.random.default_rng(random_seed)
    data = sigma * rng.normal(size=fixed_forward_params_dict["N"]) + true_forces

    # Plot true and observed force profile
    plt.plot(z, true_forces, label="true forces", color="orange")
    plt.scatter(z, data, label="observed forces", s=10)
    plt.xlabel("depth $z$")
    plt.ylabel("$F(z)$")
    plt.legend()
    plt.show()

    # Test a the jax log likelihood and gradient, and compare to ops to make sure they work
    print("testing out... (this also compiles the jit functions so may take a few seconds)")
    test_out = jitted_logp_wrapper(data, sigma, np.array([15e3, 17e3]), *fixed_forward_params_dict.values(), *not_inferred_forward_params_dict.values())
    test_out_grad = jitted_logp_grad_wrapper(data, sigma, np.array([15e3, 17e3]), *fixed_forward_params_dict.values(), *not_inferred_forward_params_dict.values())
    print(test_out, test_out_grad)
    test_out = logp_op(np.array([15e3, 17e3]))
    test_out_grad = logp_grad_op(np.array([15e3, 17e3]))
    print(test_out.eval(), test_out_grad.eval())
    print("done testing out")

    # WORKED DOWN TO HERE!
    
    #%% --------------------------------------------------------------------------------------------
    # Step 4: Define the inference model and helper functions.
    # A sensitivity analysis determined gamma_d, beta and shaft_friction limit were the most important parameters. I will additionally infer N_q which is coupled to shaft_friction_limit in a way.
    # But to start with just gamma_d to demo.

    def wrapped_op_log_likelihood(observed, *priors):
        return logp_op(*priors)
    
    # generates a sample from the forward model given the model parameters.
    def random_f(*priors, rng=None, size=None):
        rearranged = reorder_params(*priors, *fixed_forward_params_dict.values(), *not_inferred_forward_params_dict.values())
        forward = jitted_forward(*rearranged)
        return rng.normal(loc=forward, scale=sigma, size=size)

    # Use PyMC to sample from log-likelihood
    model = pm.Model()
    with model:

        # Priors, i.e. p(theta)
        priors = {}
        for name, dist_params in inferred_forward_params_dict.items():
            dist = dist_params["dist"]
            mean = dist_params["mean"]
            stdev = dist_params["stdev"]
            wrapper_fun = dist_params["wrapper_fun"]

            # Create a PyMC variable with the same name
            priors[name] = dist(name, shape=mean.shape, **wrapper_fun(mean, stdev))

        # Likelihood, i.e. p(x|theta)
        # In order to do prior/posterior predictive checks, it needs the 'random' argument which allows is to generate one sample from the pdf
        # Note, observed CANNOT be none, otherwise pymc will treat the CustomDist as a prior!
        likelihood = pm.CustomDist(
            "likelihood", *priors.values(), observed=data, logp=wrapped_op_log_likelihood, random=random_f
        )

    # Visualise the model
    pm.model_to_graphviz(model)

    initial_point = model.initial_point()
    print(initial_point)
    print(model.point_logps(initial_point))

    #%% --------------------------------------------------------------------------------------------
    # Step 4: Prior predictive check

    pp_draws = 100
    print(f"Sampling prior predictive for {pp_draws} draws...")
    with model:
        idata = pm.sample_prior_predictive(draws=pp_draws, random_seed=random_seed)
    print("Done sampling prior predictive")

    #%%

    # plot prior predictive distribution
    figsize = (10,5)
    fig, ax = plt.subplots(figsize=figsize)

    plot_xY(z, idata.prior_predictive["likelihood"], ax, label="prior predictive")
    ax.scatter(z, data, label="observed", alpha=0.6, zorder=100)
    ax.set(title="$F(z)$, Prior predictive distribution")
    plt.legend()
    plt.show()

    #%% --------------------------------------------------------------------------------------------
    # Step 5: Inference and MCMC diagnostics

    inference_draws = 100
    inference_tune = 100
    with model:
        # Use custom number of draws to replace the HMC based defaults
        # post_trace = pm.sample(draws=inference_draws, tune=inference_tune, cores=4, chains=4, random_seed=random_seed)
        post_trace = pm.sample(draws=inference_draws, tune=inference_tune, cores=1, chains=1, keep_warning_stat=True)
        # idata.extend(post_trace)
        idata = post_trace
        
    #%%

    # plot the traces
    az.plot_trace(idata, var_names=["l_gamma_d"])
    plt.show()

    #%% --------------------------------------------------------------------------------------------
    # Step 6: Posterior checks

    # Compare (plot) the prior and posterior distributions.
    # NOTE, the prior will be the sampled prior for the predictive distribution and does not reflect the actual prior used in inference.
    az.plot_dist_comparison(idata, var_names=["l_gamma_d"])

    # Posterior predictive check
    with model:
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=random_seed))

    #%%

    # plot posterior predictive distribution
    figsize = (10,5)
    fig, ax = plt.subplots(figsize=figsize)

    plot_xY(z, idata.posterior_predictive["likelihood"], ax, label="posterior predictive")
    ax.scatter(z, data, label="observed", alpha=0.6, zorder=100)
    ax.set(title="$F(z)$, Posterior predictive distribution")
    plt.legend()
    plt.show()

    # fig, ax = plt.subplots(figsize=figsize)

    # az.plot_hdi(z, idata.posterior_predictive["obs"]/1000, hdi_prob=0.5, smooth=False)
    # az.plot_hdi(z, idata.posterior_predictive["obs"]/1000, hdi_prob=0.95, smooth=False)
    # ax.scatter(z, F_noisy/1000, label="observed", alpha=0.6)
    # ax.set(title="$F(z)$, Posterior predictive distribution")
    # ax.set_xlabel("depth $z$")
    # ax.set_ylabel("$F(z) (kN)$")
    # plt.legend()
    # plt.show()

    #%%
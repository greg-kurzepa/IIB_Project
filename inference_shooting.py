## NOTES
## CURRENTLY NO SUPPORT FOR INFERRING SIGMA.
# I will have to think carefully about how to implement this. Sigma is only used in the likelihood function
# and not the forward function, so it shouldn't go in the forward params dicts.
# It also shouldn't always be passed into the grad op because that causes errors when it is fixed
# although might be worth trying that again, it could have been a bug in my code.

#%%
import graphviz
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import numpy as np
import time
import os
import logging

import packaged._model_ivp as _model_ivp

# jax.config.update('jax_enable_x64', True) #ESSENTIAL

start_time = time.strftime('%Y%m%d-%H%M%S')
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"logs\\log_{__name__}_{start_time}.log", encoding="utf-8", level=logging.DEBUG)

mpl_logger = logging.getLogger(matplotlib.__name__)
mpl_logger.setLevel(logging.WARNING)
mpl_logger = logging.getLogger(graphviz.__name__)
mpl_logger.setLevel(logging.WARNING)

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

    # Misc variables to do with runtime
    # counter = 0
    do_print = True
    do_plots = True
    prior_predictive_draws = 20

    # These parameters are never inferred
    N=200
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
        res = _model_ivp.shooting_api_wrapper(*forward_params)
        forces = res[2]
        return forces

    def forward_log_likelihood(data, sigma, *forward_params):
        # Assuming additive gaussian white noise
        # If results are all np.nan, reflecting invalid parameter combination, set probability to zero
        forces = forward_model(*forward_params)
    
        if np.all(np.isnan(forces)):
            return -np.inf
        else:
            return (-np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((data - forces) / sigma) ** 2).sum()

    def sort_x_by_y(x, y):
        return [X for (Y,X) in sorted(zip(y,x), key=lambda pair: pair[0])]

    def reorder_params(*unordered_forward_params, unordered_argnames=wrapper_arg_order, ordered_argnames=forward_arg_order):
        # Reorder unordered_forward_params in the same way that unordered_argnames would be reordered to form ordered_argnames
        unordered_idx_in_ordered = [ordered_argnames.index(x) for x in unordered_argnames]
        ordered_forward_params = sort_x_by_y(unordered_forward_params, unordered_idx_in_ordered)
        return ordered_forward_params
    
    def logp_wrapper(data, sigma, *unordered_forward_params):
        return forward_log_likelihood(data, sigma, *reorder_params(*unordered_forward_params))

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

            if do_print: print("New perform")
            logger.debug(f"\nNew Perform")

            log_likelihood = logp_wrapper(data, sigma, *inputs, *fixed_forward_params_dict.values(), *not_inferred_forward_params_dict.values())

            # Print out the log likelihood and input inferrable parameters
            if do_print:
                print(f"Printing log likelihood: {log_likelihood}")
                print(f"Printing sampled parameters")
            logger.debug(f"Log likelihood: {log_likelihood}")
            for i, inp in enumerate(inputs):
                param_name = list(inferred_forward_params_dict.keys())[i]
                logger.debug(f"i: {i}, {param_name}: {inp}")
                if do_print: print(f"i: {i}, {param_name}: {inp}")

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            outputs[0][0] = np.asarray(log_likelihood, dtype=node.outputs[0].dtype)

    # create our Ops
    logp_op = LogLikelihood()

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
    test_out = logp_wrapper(data, sigma, np.array([15e3, 17e3]), *fixed_forward_params_dict.values(), *not_inferred_forward_params_dict.values())
    print(f"test out logp original: {test_out}")
    test_out = logp_op(np.array([15e3, 17e3]))
    print(f"test out logp: {test_out.eval()}")
    print("done testing out")
    
    #%% --------------------------------------------------------------------------------------------
    # Step 4: Define the inference model and helper functions.
    # A sensitivity analysis determined gamma_d, beta and shaft_friction limit were the most important parameters. I will additionally infer N_q which is coupled to shaft_friction_limit in a way.
    # But to start with just gamma_d to demo.

    def wrapped_op_log_likelihood(observed, *priors):
        return logp_op(*priors)
    
    # generates a sample from the forward model given the model parameters.
    def random_f(*priors, rng=None, size=None):
        rearranged = reorder_params(*priors, *fixed_forward_params_dict.values(), *not_inferred_forward_params_dict.values())
        forw = forward_model(*rearranged)
        return rng.normal(loc=forw, scale=sigma, size=size)

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
    print(f"initial point: {initial_point}")
    print(f"initial points logp: {model.point_logps(initial_point)}")

    #%% --------------------------------------------------------------------------------------------
    # Step 5: Prior predictive check

    logstr = f"Sampling prior predictive for {prior_predictive_draws} draws"
    print(logstr)
    logger.info(logstr)

    with model:
        idata = pm.sample_prior_predictive(draws=prior_predictive_draws, random_seed=random_seed)
    print("Done sampling prior predictive")

    # plot prior predictive distribution
    if do_plots:
        figsize = (10,5)
        fig, ax = plt.subplots(figsize=figsize)

        plot_xY(z, idata.prior_predictive["likelihood"], ax, label="prior predictive")
        ax.scatter(z, data, label="observed", alpha=0.6, zorder=100)
        ax.set(title="$F(z)$, Prior predictive distribution")
        plt.legend()
        plt.show()

    #%% --------------------------------------------------------------------------------------------
    # Step 6: Inference and MCMC diagnostics

    inference_draws = 20
    inference_tune = 0
    chains = 1

    logstr = f"Beginning inference for {inference_draws} draws, {inference_tune} tune, {chains} chains"
    print(logstr)
    logger.info(logstr)

    with model:
        # Use custom number of draws to replace the HMC based defaults
        # post_trace = pm.sample(draws=inference_draws, tune=inference_tune, cores=4, chains=4, random_seed=random_seed)
        post_trace = pm.sample(draws=inference_draws, tune=inference_tune, cores=min(chains, 4), chains=chains, keep_warning_stat=True)
        idata.extend(post_trace)
        # idata = post_trace
    # print(f"Counter was: {counter}")
        
    #%%

    # plot the traces
    if do_plots:
        az.plot_trace(idata, var_names=[key for key in inferred_forward_params_dict.keys()])
        plt.show()

        # Compare (plot) the prior and posterior distributions.
        # NOTE, the prior will be the sampled prior for the predictive distribution and does not reflect the actual prior used in inference.
        az.plot_dist_comparison(idata, var_names=[key for key in inferred_forward_params_dict.keys()])
        plt.show()

    #%% --------------------------------------------------------------------------------------------
    # Step 7: Posterior checks

    logstr = f"""Sampling posterior predictive"""
    print(logstr)
    logger.info(logstr)

    # Posterior predictive check
    with model:
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=random_seed))

    # plot posterior predictive distribution
    if do_plots:
        figsize = (10,5)
        fig, ax = plt.subplots(figsize=figsize)

        plot_xY(z, idata.posterior_predictive["likelihood"], ax, label="posterior predictive")
        ax.scatter(z, data, label="observed", alpha=0.6, zorder=100)
        ax.set(title="$F(z)$, Posterior predictive distribution")
        plt.legend()
        plt.show()

    #%% Step 8: Save results

    save_dir = r"C:\Users\gregk\Documents\MyDocuments\IIB\Project\Alt Code\Model - Shooting Tests\saved_inferences"
    save_name = f"{start_time}.nc"
    idata.to_netcdf(os.path.join(save_dir, save_name))

    #%%
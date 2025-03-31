#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import copy
import multiprocessing

import packaged._pile_and_soil as _pile_and_soil
import packaged._model_springs as _model_springs

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

    #%% --------------------------------------------------------------------------------------------
    # Step 0: Define the likelihood functions and the black box Op.

    def forward_model(pile, soil, P, z_w, N=100, t_res_clay=0.9):
        res = _model_springs.solve_springs4(pile, soil, P, z_w, N=N, t_res_clay=t_res_clay, tau_over_tau_ult_func = None, Q_over_Q_ult_func = None)
        return res[0]

    def log_likelihood_f(sigma, data, *forward_params):
        # assuming additive gaussian white noise
        logp = - np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((data - forward_model(*forward_params)) / sigma) ** 2
        return logp.sum()

    def soil_from_pymc_params(gamma_d):
        soil = copy.deepcopy(soil_true)

        for layer_i in range(gamma_d.shape[0]):
            soil.layers[layer_i].gamma_d = gamma_d[layer_i]

        return soil

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

            soil = soil_from_pymc_params(gamma_d)

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
    soil_true = _pile_and_soil.Soil([layer1, layer2])

    # these parameters are known to the model
    sigma = 0.05e6 # standard deviation of noise
    z_w = 3 # water table depth

    truemodel = forward_model(pile, soil_true, P, z_w, N)

    # make data
    random_seed = 716743
    rng = np.random.default_rng(random_seed)
    data = sigma * rng.normal(size=N) + truemodel

    # print("testing out...")
    # test_out = loglike_op(data, sigma, np.array([15e3, 18.5e3]))
    # print(test_out.eval())
    # print("done testing out")

    #%% --------------------------------------------------------------------------------------------
    # Step 2: Define the inference model and the priors on the parameters.
    # A sensitivity analysis determined gamma_d, beta and shaft_friction limit were the most important parameters. I will additionally infer N_q which is coupled to shaft_friction_limit in a way.
    # But to start with just gamma_d.
    # Assume we know the rough category of soil e.g. loose or dense sand

    def wrapped_log_likelihood(data, sigma, gamma_d):
        # data, or observed is always passed as the first input of CustomDist
        return loglike_op(data, sigma, gamma_d)

    def random_f(sigma, gamma_d, rng=None, size=None):
        # generates a sample from the forward model given the model parameters.
        # this needs to generate the random additive noise

        soil = soil_from_pymc_params(gamma_d)

        return rng.normal(loc=forward_model(pile, soil, P, z_w, N), scale=sigma, size=size)

    def get_lognormal_params(mean, stdev):
        # The mu, sigma parameters of a lognormal distribution are not the true mean and standard deviation of the distribution.
        # This function takes the true mean and standard deviation and outputs the mu, sigma parameters of the lognormal distribution.
        mu_lognormal = np.log(mean**2 / np.sqrt(stdev**2 + mean**2))
        sigma_lognormal = np.sqrt(np.log(1 + (stdev**2 / mean**2)))
        return {"mu" : mu_lognormal, "sigma" : sigma_lognormal}

    # Priors for each layer
    gamma_d_mean = np.array([15e3, 18.5e3])
    gamma_d_stdev = np.array([3e3, 3e3])

    # use PyMC to sampler from log-likelihood
    model = pm.Model()
    with model:

        # priors, i.e. p(theta)
        gamma_d = pm.LogNormal("gamma_d", shape=2, **get_lognormal_params(gamma_d_mean, gamma_d_stdev))

        # likelihood, i.e. p(x|theta)
        # in order to do prior/posterior predictive checks, it needs the 'random' argument which allows is to generate one sample from the pdf
        forces = pm.CustomDist(
            "forces", sigma, gamma_d, observed=data, logp=wrapped_log_likelihood, random=random_f
        )

    # Visualise the model
    pm.model_to_graphviz(model)

    #%% --------------------------------------------------------------------------------------------
    # Step 4: Prior predictive check

    pp_draws = 100
    print(f"Sampling prior predictive for {pp_draws} draws...")
    with model:
        idata = pm.sample_prior_predictive(draws=pp_draws, random_seed=random_seed)
    print("Done sampling prior predictive")

    # plot prior predictive distribution
    figsize = (10,5)
    fig, ax = plt.subplots(figsize=figsize)

    plot_xY(z, idata.prior_predictive["forces"], ax, label="prior predictive")
    ax.scatter(z, data, label="observed", alpha=0.6, zorder=100)
    ax.set(title="$F(z)$, Prior predictive distribution")
    plt.legend()
    plt.show()

    # az.plot_ppc(idata, group="prior")
    # plt.show()

    #%% --------------------------------------------------------------------------------------------
    # Step 5: Inference and MCMC diagnostics

    inference_draws = 500
    inference_tune = 500
    with model:
        # Use custom number of draws to replace the HMC based defaults
        # post_trace = pm.sample(draws=inference_draws, tune=inference_tune, cores=4, chains=4, random_seed=random_seed)
        post_trace = pm.sample(inference_draws, cores=4, chains=4)
        idata.extend(post_trace)

    # plot the traces
    az.plot_trace(idata, var_names=["gamma_d"])
    plt.show()

    #%% --------------------------------------------------------------------------------------------
    # Step 6: Posterior checks

    # Compare (plot) the prior and posterior distributions.
    # NOTE, the prior will be the sampled prior for the predictive distribution and does not reflect the actual prior used in inference.
    az.plot_dist_comparison(idata, var_names=["gamma_d"])

    # Posterior predictive check
    with model:
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=random_seed))

    #%%

    # plot posterior predictive distribution
    figsize = (10,5)
    fig, ax = plt.subplots(figsize=figsize)

    plot_xY(z, idata.posterior_predictive["forces"], ax, label="posterior predictive")
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
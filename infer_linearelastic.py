#%%
import graphviz
import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import time
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import linearelastic.solver as solver
from packaged._inference import plot_profile, plot_idata_trace, _plot_xY, _get_lognormal_params
from packaged._model_springs import SolveData

if __name__ == "__main__":

    print(f"Running on PyMC v{pm.__version__}")
    az.style.use("arviz-darkgrid")

    P = 3.6e6
    N = 100
    sigma = 20e-6

    pile = solver.Pile(R=0.3, L=30.0, E=35e9)
    E_soil_list = np.array([0.2e7, 2e7])
    base_depths = np.array([12.5, pile.L])
    Kb = 600e6

    lambda_k_list = np.sqrt(pile.C * E_soil_list / (pile.A * pile.E))
    Omega = Kb / (lambda_k_list[1] * pile.E * pile.A)
    z = np.linspace(0, pile.L, N)

    # Generate/load synthetic data
    random_seed = 716743
    rng = np.random.default_rng(random_seed)

    # d, strain = solver.solve_linearelastic_analytic(z, P, pile, lambda_k_list, Omega, base_depths)
    # force = - pile.A * pile.E * strain
    # data = force + sigma * rng.normal(size=N)
    df_data = pd.read_csv(r"C:\Users\gregk\Documents\MyDocuments\IIB\Project\Alt Code\Model - Concrete Cracking\observed\compression-4,6MN-strain.csv")
    true_strain = np.array(df_data["True Strain"])
    data = np.array(df_data["Observed Strain"])

    plt.plot(z, true_strain, label="True Strain")
    plt.scatter(z, data, label="Observed Strain", color="red", s=10)
    plt.grid()
    plt.show()
    #%%

    def f_log_likelihood(data, E_soil_list, Kb):
        lambda_k_list = np.sqrt(pile.C * E_soil_list / (pile.A * pile.E))
        Omega = Kb / (lambda_k_list[1] * pile.E * pile.A)

        # Solve the problem
        # print("lk, Om: ", lambda_k_list, Omega)
        d, gradient = solver.solve_linearelastic_analytic(z, P, pile, lambda_k_list, Omega, base_depths)
        strain = -gradient

        # Compute the log likelihood
        log_likelihood = (-np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((data - strain) / sigma) ** 2).sum()

        return log_likelihood

    def random_f(E_soil_list, Kb, rng=None, size=None):
        lambda_k_list = np.sqrt(pile.C * E_soil_list / (pile.A * pile.E))
        Omega = Kb / (lambda_k_list[1] * pile.E * pile.A)

        # Solve the problem
        d, gradient = solver.solve_linearelastic_analytic(z, P, pile, lambda_k_list, Omega, base_depths)
        strain = -gradient
        noisy_strain = rng.normal(loc=strain, scale=sigma, size=size)

        return noisy_strain

    # Define a pytensor Op for our likelihood function
    class LogLikelihood(Op):
        # The inputs are parameters to be inferred. The other parameters are used from the global values.
        def make_node(self, data, E_soil_list, Kb) -> Apply:
            # Convert inputs to tensor variables
            # Note the difference with black box op: pt.as_tensor_variable instead of pt.as_tensor
            inputs = [pt.as_tensor_variable(data),
                    pt.as_tensor_variable(E_soil_list),
                    pt.as_tensor_variable(Kb)]

            # Define output type, in this case a single scalar
            outputs = [pt.scalar()]

            # Apply is an object that combines inputs, outputs and an Op (self)
            return Apply(self, inputs, outputs)

        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            # This is the method that compute numerical output given numerical inputs.

            log_likelihood = f_log_likelihood(*inputs)

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            outputs[0][0] = np.asarray(log_likelihood, dtype=node.outputs[0].dtype)

    logp_op = LogLikelihood()

    # PyMC model
    model = pm.Model()
    with model:
        # m_E_soil_list = pm.LogNormal("_E_soil", **_get_lognormal_params(mean=E_soil_list, stdev=E_soil_list), shape=2)
        # m_Kb = pm.LogNormal("_Kb", **_get_lognormal_params(mean=Kb, stdev=10*Kb))
        m_E_soil_list = pm.Uniform("_E_soil", lower=[1e5, 1e5], upper=[1000e6, 1000e6], shape=2)
        m_Kb = pm.Uniform("_Kb", lower=1e5, upper=1000e6)

        likelihood = pm.CustomDist(
            "likelihood", m_E_soil_list, m_Kb, observed=data, logp=logp_op, random=random_f
        )

    # Visualise the model
    pm.model_to_graphviz(model)

    out_log_likelihood = f_log_likelihood(data, E_soil_list, Kb)
    out_logp_op = logp_op(data, E_soil_list, Kb).eval()
    print(f"Forward out: {out_log_likelihood}, Op out: {out_logp_op}")

    initial_point = model.initial_point()
    print(f"Initial point: {initial_point}")
    print(f"Initial point logp: {model.point_logps(initial_point)}")

    #%% Prior predictive check

    draws = 5000
    print(f"Sampling prior predictive for {draws} draws...")
    with model:
        idata_pp = pm.sample_prior_predictive(draws=draws, random_seed=random_seed)
    print("Done sampling prior predictive")

    #%%

    plot_idata_trace(z, idata_pp["prior_predictive"]["likelihood"], data, title="Strain Prior Predictive Distribution")

    # %%

    inference_draws = 5000
    chains = 4
    print(f"Sampling SMC, {inference_draws} draws...")
    with model:
        idata_post_smc = pm.sample_smc(draws=inference_draws, cores=min(chains, 4), chains=4, random_seed=random_seed)
    print("Done sampling")

    do_plot = True
    if do_plot:
        var_names = [x.name for x in model.free_RVs]

        az.plot_trace(idata_post_smc, var_names=var_names)
        plt.show()

        # Compare (plot) the prior and posterior distributions.
        # NOTE, the prior will be the sampled prior for the predictive distribution and does not reflect the actual prior used in inference.
        az.plot_dist_comparison(idata_post_smc, var_names=var_names)
        plt.show()

    #%%

    print("Sampling posterior predictive...")
    with model:
        idata_postp = pm.sample_posterior_predictive(idata_post_smc, random_seed=random_seed)
    print("Done sampling posterior predictive")

    #%%

    plot_idata_trace(z, idata_postp["posterior_predictive"]["likelihood"], data, title="Strain Posterior Predictive Distribution")

    #%% Save the idata, and the model

    import cloudpickle
    save_name = "results\\linearelastic_idata.pkl"
    with open(save_name, "wb") as f:
        cloudpickle.dump(idata_pp, f)

    #%%

    model_save_name = "results\\linearelastic_model.pkl"
    with open(model_save_name, "wb") as f:
        cloudpickle.dump(model, f)

    # %% read it back just to check
    with open(model_save_name, "rb") as f:
        model_loaded = cloudpickle.load(f)

    with open(save_name, "rb") as f:
        idata_loaded = cloudpickle.load(f)
# %%

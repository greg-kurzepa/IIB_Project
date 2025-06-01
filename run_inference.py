#%%
import numpy as np
import packaged._inference as inference
import packaged._utilities as utilities
import pymc as pm

if __name__ == "__main__":
    config = inference.InferenceConfig(
        P = 4.6e6
    )
        # inferred_forward_params_dict = {
        #     "l_gamma_d" : {
        #         "dist" : pm.LogNormal,
        #         "wrapper_fun" : inference._get_lognormal_params,
        #         "args" : {
        #             "mean" : np.array([14.5e3, 18e3]),
        #             "stdev" : np.array([3e3, 3e3]),
        #         }
        #     },
        #     "l_e" : {
        #         "dist" : pm.TruncatedNormal,
        #         "wrapper_fun" : None,
        #         "args" : {
        #             "mu" : np.array([0.8, 0.45]),
        #             "sigma" : np.array([0.2, 0.2]),
        #             "lower" : 0.01,
        #             "upper" : 1.0,
        #         }
        #     },
        #     "l_c2" : { # beta
        #         "dist" : pm.LogNormal,
        #         "wrapper_fun" : inference._get_lognormal_params,
        #         "args" : {
        #             "mean" : 1.25*np.array([0.214, 0.46]),
        #             "stdev" : 1.25*np.array([0.08, 0.08]),
        #         }
        #     },
        #     "l_shaft_pressure_limit" : {
        #         "dist" : pm.LogNormal,
        #         "wrapper_fun" : inference._get_lognormal_params,
        #         "args" : {
        #             "mean" : np.array([47.8e3, 96e3]),
        #             "stdev" : np.array([20e3, 20e3]),
        #         }
        #     }
        # }
    # )

    data_dir = r"C:\Users\gregk\Documents\MyDocuments\IIB\Project\Alt Code\Model - Concrete Cracking\observed\compression-4,6MN-strain_corrected.csv"
    m1 = inference.make_pymc_model(solver_type = "scipy_fsolve_bvp", data_dir=data_dir, inference_config=config)

    #%%

    # plot an example profile to verify the deterministic function works
    # prior = (np.array([15000, 17000]), np.array([0.8, 0.45]), 1.25*np.array([0.214, 0.46]), np.array([47.8e3, 96e3]))
    prior = (np.array([15000, 17000]),)
    _ = inference.plot_profile(m1, prior)

    #%% prior predictive
    idata_prior = inference.prior_predictive(m1.model, draws=1000)

    #%%
    utilities.plot_prior_samples(idata_prior)
    inference.plot_idata_trace(z=m1.config.z, idata_trace=idata_prior["prior_predictive"]["likelihood"], data=m1.data)

    #%% sample
    idata_posterior = inference.sample_posterior(m1.model, inference_draws=2000, do_plot=True)

    #%% posterior predictive, extends idata_posterior
    inference.posterior_predictive(m1.model, idata_posterior)

    #%%
    inference.plot_idata_trace(z=m1.config.z, idata_trace=idata_posterior["posterior_predictive"]["likelihood"], data=m1.data)

    #%% Save the idata, and the model
    idata_prior.extend(idata_posterior)

    import cloudpickle
    save_name = "results\\h_idata.pkl"
    with open(save_name, "wb") as f:
        cloudpickle.dump(idata_prior, f)

    model_save_name = "results\\h_model.pkl"
    with open(model_save_name, "wb") as f:
        cloudpickle.dump(m1.model, f)

# %%

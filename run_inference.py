#%%
import numpy as np
import packaged._inference as inference
import packaged._utilities as utilities

if __name__ == "__main__":
    config = inference.InferenceConfig()
    config.fixed_forward_params_dict["P"] = 6.2e6

    # m1 = inference.make_pymc_model(model_type = "jax_fsolve")
    m1 = inference.make_pymc_model(solver_type = "scipy_fsolve", inference_config=config)

    # plot an example profile to verify the deterministic function works
    prior = np.array([15000,17000])
    _ = inference.plot_profile(m1, (prior,))

    #%% prior predictive
    idata = inference.prior_predictive(m1.model, draws=1000)

    #%%
    utilities.plot_prior_samples(idata)

    #%%
    inference.plot_idata_trace(z=m1.config.z, idata_trace=idata["prior_predictive"]["likelihood"], data=m1.data)

    #%% sample
    inference.sample_posterior(m1.model, do_plot=True)

    #%% posterior predictive
    inference.posterior_predictive(m1.model, idata)

# %%

#%%
import packaged._inference as inference

if __name__ == "__main__":
    model, config, data = inference.make_pymc_model(model_type = "jax_fsolve")

    #%% prior predictive
    idata = inference.prior_predictive(model, draws=100)
    inference.plot_idata_trace(z=config.z, idata_trace=idata["prior_predictive"]["likelihood"], data=data)

    #%% sample
    inference.sample(model, do_plot=True)

    #%% posterior predictive
    inference.posterior_predictive(model, idata)

# %%

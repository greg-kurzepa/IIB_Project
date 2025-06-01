## NOTES
## CURRENTLY NO SUPPORT FOR INFERRING SIGMA.
# I will have to think carefully about how to implement this. Sigma is only used in the likelihood function
# and not the forward function, so it shouldn't go in the forward params dicts.
# It also shouldn't always be passed into the grad op because that causes errors when it is fixed
# although might be worth trying that again, it could have been a bug in my code.

#%%
import graphviz
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
# import jax
# import jax.numpy as jnp
import time
import os
import logging
import matplotlib
import matplotlib.pyplot as plt

from . import _utilities
# from . import _ops_jax
from . import _ops_scipy

# This wrapper is necessary to allow pm.sample to work with multiple cores.
# if __name__ == "__main__":

print(f"Running on PyMC v{pm.__version__}")
az.style.use("arviz-darkgrid")

# Jax was written primarily for neural networks, which do not need high precision.
# Here it's used for solving a stiff ODE so 64-bit accuracy is ESSENTIAL.
# jax.config.update('jax_enable_x64', True)

# Set up logging to ease debugging
start_time = time.strftime('%Y%m%d-%H%M%S')
filepath = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(filepath, "logs", f"log_{__name__}_{start_time}.log"), encoding="utf-8", level=logging.DEBUG)
# Stop 3rd party libraries from clogging up the logs
mpl_logger = logging.getLogger(matplotlib.__name__)
mpl_logger.setLevel(logging.WARNING)
mpl_logger = logging.getLogger(graphviz.__name__)
mpl_logger.setLevel(logging.WARNING)
# mpl_logger = logging.getLogger(jax.__name__)
# mpl_logger.setLevel(logging.WARNING)

default_random_seed = 716743

# this must be the same for all solvers
# NOTE the jax solver is out of date in this regard, so it cannot be used right now.
# forward_arg_order = ("pile_D", "pile_L", "f_ck", "alpha_e", "G_F0", "reinforcement_ratio",
#                     "l_layer_type", "l_gamma_d", "l_e", "l_c1", "l_c2", "l_shaft_pressure_limit", "l_end_pressure_limit",
#                     "l_base_depth", "P", "z_w", "N", "t_res_clay")
# static_argnames = ("pile_L", "f_ck", "alpha_e", "G_F0", "reinforcement_ratio", "P", "z_w", "N", "t_res_clay")
forward_arg_order = ("pile_D", "pile_L", "pile_E", "l_layer_type", "l_gamma_d", "l_e", "l_c1", "l_c2", "l_shaft_pressure_limit", "l_end_pressure_limit", "l_base_depth", "P", "z_w", "N", "t_res_clay")
static_argnames = ("pile_L", "pile_E", "P", "z_w", "N", "t_res_clay")
forward_static_argnums = [forward_arg_order.index(x) for x in static_argnames]

# Function to enable plotting functional confidence intervals, taken from pytorch examples
def _plot_xY(x, Y, ax, label=None):
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
def _get_lognormal_params(mean, stdev):
    mu_lognormal = np.log(mean**2 / np.sqrt(stdev**2 + mean**2))
    sigma_lognormal = np.sqrt(np.log(1 + (stdev**2 / mean**2)))
    return {"mu" : mu_lognormal, "sigma" : sigma_lognormal}

class InferenceConfig():
    """This class mainly specifies the physical pile and soil parameters, including the ground truth as well as
    which parameters are to be inferred and what their priors are. It also contains other parameters relevant
    to inference and runtime.

    Members:
        - `do_print` (bool): Whether debugging information is printed to the console (it is always saved in logs).
        - `do_plots` (bool): Whether plots of results are to be shown including prior predictive, parameter posterior and posterior predictive distributions.
        - `sigma` (float): Standard deviation of noise that is added to the deterministic physical function.
        - `sand_plug_factor`: From the API2GEO documentation, this parameter increases Beta of a plugged pile in sand.
        - `fixed_forward_params_dict`: Non-inferred physical model parameters.
        - `inferrable_forward_params_dict`: Physical model parameters that COULD be inferred. It specifies their ground truth values. The inference model only sees these if they are not specified as a prior in inferred_forward_params_dict.
        - `inferred_forward_params_dict`: Inferred physical model parameters, must be selected from in inferred_model_params_dict. It specifies their prior distributions.
    """

    def __init__(self, sand_plug_factor: float = 1.25, P: float = 3.6e6, N_layers: int = 2,
                 fixed_forward_params_dict: dict = None,
                 inferred_forward_params_dict: dict = None,
                 inferrable_forward_params_dict: dict = None):
        
        self.do_print = False
        self.do_plot = True

        self.N_layers = N_layers
        self.sand_plug_factor = sand_plug_factor
        
        # These parameters are never inferred
        if fixed_forward_params_dict is None:
            N = 100
            self.fixed_forward_params_dict = {
                "P" : P, # top axial load
                "N" : N, # number of nodes along pile
                "z_w" : 3, # water table depth
                "l_layer_type" : np.array([1, 1]), # 0 for clay, 1 for sand
                "l_base_depth" : np.array([12.5, 30]), # base depth of each layer (final value is the length of the pile)
                "pile_L" : 30, # pile length
                "pile_D" : np.full(N, 0.6), # pile diameter
                "pile_E" : 35e9, # pile elastic modulus
                "t_res_clay" : 0.9
            }
            
            # self.fixed_forward_params_dict = {
            #     "P" : 1.8e6, # top axial load
            #     "N" : N, # number of nodes along pile
            #     "z_w" : 3, # water table depth
            #     "l_layer_type" : np.array([1, 1]), # 0 for clay, 1 for sand
            #     "l_base_depth" : np.array([12.5, 30]), # base depth of each layer (final value is the length of the pile)
            #     "t_res_clay" : 0.9,

            #     "pile_L" : 30, # pile length
            #     "pile_D" : np.full(N, 0.6), # pile diameter
            #     "f_ck" : 50, # concrete compressive strength, in MPa
            #     "alpha_e" : 1.0, # scaling factor for concrete elastic modulus based on type of aggregate
            #     "G_F0" : 0.065, # fracture energy of concrete, in N/mm
            #     "reinforcement_ratio" : 0.04, # steel area as proportion of pile area
            # }
        else:
            self.fixed_forward_params_dict = fixed_forward_params_dict

        # Sigma is only used in the log likelihood and not the forward model, so is left out of the below dictionaries
        self.sigma = 20e-6 # 20 microstrains

        # Contains the ground truth values of the parameters for each layer that *could* be inferred
        # c1 is N_c for clay and N_q for sand
        # c2 is beta for sand and psi for clay
        # shaft_friction_limit and end_bearing_limit are only ever inferred for sand. for clay, they are always inf.
        if inferrable_forward_params_dict is None:
            self.inferrable_forward_params_dict = {
            "l_gamma_d" : np.array([15e3, 17e3]),
            "l_e" : np.array([0.689, 0.441]),
            "l_c1" : np.array([8, 40]),
            "l_c2" : np.array([sand_plug_factor*0.214, sand_plug_factor*0.46]),
            "l_shaft_pressure_limit" : np.array([47.8e3, 96e3]),
            "l_end_pressure_limit" : np.array([1.9e6, 10e6]),
        }
        else:
            self.inferrable_forward_params_dict = inferrable_forward_params_dict

        self.z = np.linspace(0, self.fixed_forward_params_dict["pile_L"], self.fixed_forward_params_dict["N"])
        self.dz = self.z[1] - self.z[0] # length of one element

        # Contains the parameters to be inferred and their prior distributions
        # wrapper_fun takes mean, stdev as input and returns the parameters to input to dist
        # Prior widths I set here reflect knowledge the rough category of soil e.g. loose or dense sand
        if inferred_forward_params_dict is None:
            self.inferred_forward_params_dict = {
                "l_gamma_d" : {
                    "dist" : pm.LogNormal,
                    "wrapper_fun" : _get_lognormal_params,
                    "args" : {
                        "mean" : np.array([14.5e3, 18e3]),
                        "stdev" : np.array([3e3, 3e3]),
                    }
                }
            }
        else:
            self.inferred_forward_params_dict = inferred_forward_params_dict

        self.not_inferred_forward_params_dict = {key : self.inferrable_forward_params_dict[key] for key in self.inferrable_forward_params_dict if key not in self.inferred_forward_params_dict.keys()}

        # In the actual code, these parameters will be passed around as tuples, not dicts.
        # This is because jax.grad() only accepts positional arguments and writing a wrapper that takes account of the argnums argument was painful.
        # The below members are solely for the purpose of dealing with this nicely. The below code is ugly but makes the rest of the code easy to understand.

        # likelihood_arg_order is the order to pass into the *likelihood* function
        self.likelihood_arg_order = ("sigma", "data") + forward_arg_order
        self.likelihood_static_argnums = [self.likelihood_arg_order.index(x) for x in static_argnames]
        self.grad_argnums = [self.likelihood_arg_order.index(key) for key in self.inferred_forward_params_dict.keys()]

        # wrapper_arg_order is the order to pass into the *wrapped* likelihood & grad functions, excluding data, sigma which are always first
        self.wrapper_arg_order = (*self.inferred_forward_params_dict.keys(), *self.fixed_forward_params_dict.keys(), *self.not_inferred_forward_params_dict.keys())

        # When using the Ops, only the inferred parameters are passed for simplicity, in the following order:
        # op_arg_order = ("data", "sigma", *inferred_forward_params_dict.keys())

        # When using ground truth into the forward function (skipping the likelihood function), we use inferrable parameters.
        self.truth_arg_order = (*self.inferrable_forward_params_dict.keys(), *self.fixed_forward_params_dict.keys())

# generates a sample from the forward model given the model parameters.
def random_f(*priors, rng=None, size=None, noise=True, config=None, forward=None):
    if forward is None or config is None:
        raise ValueError("forward and config must be provided")

    rearranged = _utilities.reorder_params(*priors, *config.fixed_forward_params_dict.values(), *config.not_inferred_forward_params_dict.values(), config=config)
    forward_eval = forward(*rearranged)

    if noise:  
        return rng.normal(loc=forward_eval, scale=config.sigma, size=size)
    else:
        return forward_eval
    
class MadeModel():
    def __init__(self, model: pm.Model, config: InferenceConfig, data: np.ndarray, 
                 forward: callable = None):
        self.model = model
        self.config = config
        self.data = data
        self.forward = forward

def make_pymc_model(solver_type: str = "scipy_fsolve_simultaneous", inference_config: InferenceConfig = None, data_dir: str = None, random_seed: int = default_random_seed):
    solver_type = solver_type

    if inference_config is None:
        config = InferenceConfig()
    else:
        config = inference_config

    # Of the code below, the rest of the program will only interact with logp_op and (if it exists) logp_grad_op
    # The jax options have the grad ops, allowing HMC monte carlo, but the scipy ones do not so require Metropolis sampling
    if solver_type == "jax_fsolve":
        raise NotImplementedError()
        # forward, logp_op, logp_grad_op, test_out = _ops_jax.create_jax_ops(config)
    elif solver_type == "scipy_fsolve_simultaneous":
        forward, logp_op, _, test_out = _ops_scipy.create_scipy_ops(config, model_type="simultaneous")
    elif solver_type == "scipy_fsolve_shooting":
        forward, logp_op, _, test_out = _ops_scipy.create_scipy_ops(config, model_type="shooting")
    elif solver_type == "scipy_fsolve_bvp":
        forward, logp_op, _, test_out = _ops_scipy.create_scipy_ops(config, model_type="bvp")

    # Generate ground truth data
    random_seed = random_seed
    rng = np.random.default_rng(random_seed)

    if data_dir is not None:
        df_data = pd.read_csv(data_dir)
        true_strain = np.array(df_data["True Strain"])
        data = np.array(df_data["Observed Strain"])
    else:
        raise NotImplementedError("Still does forces not strains")
        unordered_true_params = (*config.inferrable_forward_params_dict.values(), *config.fixed_forward_params_dict.values())
        ordered_true_params = _utilities.reorder_params(*unordered_true_params, config=config, unordered_argnames=config.truth_arg_order)
        true_forces = np.array(forward(*ordered_true_params))
        data = config.sigma * rng.normal(size=config.fixed_forward_params_dict["N"]) + true_forces

    if config.do_plot:
        # Plot true and observed force profile
        plt.plot(config.z, true_strain, label="true strains", color="orange")
        plt.scatter(config.z, data, label="observed strains", s=10)
        plt.xlabel("depth $z$")
        plt.ylabel("$strain (z)$")
        plt.legend()
        plt.show()

    # This is closure to create the function that PyMC will call to generate samples from the forward model
    def random_f_wrapper(*priors, rng=None, size=None):
        return random_f(*priors, rng=rng, size=size, noise=True, config=config, forward=forward)

    # Define PyMC model
    model = pm.Model()
    with model:

        # Priors, i.e. p(theta)
        priors = {}
        for name, dist_params in config.inferred_forward_params_dict.items():
            dist = dist_params["dist"]
            wrapper_fun = dist_params["wrapper_fun"] if dist_params["wrapper_fun"] is not None else lambda **args: args
            args = dist_params["args"]

            # Create a PyMC variable with the same name
            priors[name] = dist(name, shape=config.N_layers, **wrapper_fun(**args))

        # Likelihood, i.e. p(x|theta)
        # In order to do prior/posterior predictive checks, it needs the 'random' argument which allows is to generate one sample from the pdf
        # Note, observed CANNOT be none, otherwise pymc will treat the CustomDist as a prior!
        likelihood = pm.CustomDist(
            "likelihood", *priors.values(), observed=data, logp=logp_op, random=random_f_wrapper
        )

    # Visualise the model
    pm.model_to_graphviz(model)

    # Test that the Ops give the right output
    test_out(config.sigma, data, *pm.draw(model.free_RVs, draws=1, random_seed=random_seed))

    initial_point = model.initial_point()
    print(f"Initial point: {initial_point}")
    print(f"Initial point logp: {model.point_logps(initial_point)}")

    return MadeModel(model, config, data, forward)

def prior_predictive(model: pm.Model, idata: az.InferenceData = None, draws: int = 500, random_seed: int = default_random_seed):

    print(f"Sampling prior predictive for {draws} draws...")
    with model:
        idata_pp = pm.sample_prior_predictive(draws=draws, random_seed=random_seed)
    print("Done sampling prior predictive")

    if idata is not None:
        idata.extend(idata_pp)
    else:
        return idata_pp

def sample_posterior(model: pm.Model, sample_type="smc", idata: az.InferenceData = None, inference_draws: int = 100, tune_draws: int = 100, chains: int = 4, do_plot: bool = True, random_seed: int = default_random_seed):
    
    if sample_type == "smc":
        print(f"Sampling SMC for {inference_draws} draws")
        with model:
            idata_post = pm.sample_smc(draws=inference_draws, cores=min(chains, 4), chains=4, random_seed=random_seed)
        print("Done sampling!")
    
    else:
        print(f"Sampling {inference_draws} draws with {tune_draws} tuning steps...")
        with model:
            idata_post = pm.sample_smc(draws=inference_draws, tune=tune_draws, cores=min(chains, 4), chains=4, random_seed=random_seed)
        print("Done sampling")

    if do_plot:
        try:
            var_names = [x.name for x in model.free_RVs]

            az.plot_trace(idata_post, var_names=var_names)
            plt.show()

            # Compare (plot) the prior and posterior distributions.
            # NOTE, the prior will be the sampled prior for the predictive distribution and does not reflect the actual prior used in inference.
            az.plot_dist_comparison(idata_post, var_names=var_names)
            plt.show()
        except Exception as e:
            print(f"Error while plotting posterior plots: {e}")

    if idata is not None:
        idata.extend(idata_post)
    else:
        return idata_post
    
def posterior_predictive(model: pm.Model, idata_posterior: az.InferenceData = None, random_seed: int = default_random_seed):

    print("Sampling posterior predictive...")
    with model:
        idata_posterior.extend(pm.sample_posterior_predictive(idata_posterior, random_seed=random_seed))
    print("Done sampling posterior predictive")
    
def plot_profile(made_model: MadeModel, priors: tuple = None, ax: matplotlib.axes.Axes = None):
    """Takes in priors_tuple (must be in the right order as the priors in the model)
    and plots the output of the forward function without the additive noise.

    If priors_tuple is not supplied, the prior(s) will be randomly sampled from the model."""

    show = False
    if ax is None:
        figsize = (10,5)
        fig, ax = plt.subplots(figsize=figsize)
        show = True

    if priors is None:
        priors = tuple(pm.draw(made_model.model.free_RVs, draws=1))
    else:
        if type(priors) is not tuple and type(priors) is not list:
            raise ValueError("priors must be a tuple or list")

    profile = random_f(*priors, rng=None, size=None, noise=False, config=made_model.config, forward=made_model.forward)

    ax.plot(made_model.config.z, profile, label=f"priors: {str(priors)}")
    ax.set_xlabel("depth $z$")
    ax.set_ylabel("$strain (z)$")
    ax.set_title("Deterministic Strain Profile")
    ax.set_xlim(left=0, right=made_model.config.fixed_forward_params_dict["pile_L"])

    if show:
        plt.show()

    return profile

def plot_idata_trace(z, idata_trace, data=None, ax=None, trace_label=None, title=None):

    if ax is None:
        figsize = (10,5)
        fig, ax = plt.subplots(figsize=figsize)

    _plot_xY(z, idata_trace, ax, label=trace_label)

    if data is not None:
        ax.scatter(z, data, label="observed", alpha=0.6, zorder=100)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("$Strain (z)$, Prior predictive distribution")

    if ax is not None:
        plt.legend()
        plt.show()

def save_idata(idata, dir="saved_idata"):
    save_name = f"{start_time}.nc"
    idata.to_netcdf(os.path.join(dir, save_name))
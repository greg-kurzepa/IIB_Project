import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.gradient import grad_not_implemented

from . import _utilities
from . import _model_springs
from . import _pile_and_soil
from . import _inference

def prepare_for_scipy(*forward_params):
        inp_keys = _inference.forward_arg_order
        d = {k: v for k, v in zip(inp_keys, forward_params)}

        # Define the pile
        pile = _pile_and_soil.Pile(R=d["pile_D"][0]/2,
                                   L=d["pile_L"],
                                   f_ck=d["f_ck"],
                                   alpha_e=d["alpha_e"],
                                   G_F0=d["G_F0"],
                                   reinforcement_ratio=d["reinforcement_ratio"],)

        # Define the soil
        layers = []
        for i in range(len(d["l_layer_type"])):
            if d["l_layer_type"][i] == 0: # clay
                layers.append(_pile_and_soil.ClayLayer(
                    gamma_d = d["l_gamma_d"][i],
                    e = d["l_e"][i],
                    N_c = d["l_c1"][i],
                    psi = d["l_c2"][i],
                    shaft_pressure_limit = d["l_shaft_pressure_limit"][i],
                    end_pressure_limit = d["l_end_pressure_limit"][i],
                    base_depth = d["l_base_depth"][i]
                ))
            elif d["l_layer_type"][i] == 1: # sand
                layers.append(_pile_and_soil.SandLayer(
                    gamma_d = d["l_gamma_d"][i],
                    e = d["l_e"][i],
                    N_q = d["l_c1"][i],
                    beta = d["l_c2"][i],
                    shaft_pressure_limit = d["l_shaft_pressure_limit"][i],
                    end_pressure_limit = d["l_end_pressure_limit"][i],
                    base_depth = d["l_base_depth"][i]
                ))
            else:
                raise ValueError(f"Unknown layer id '{d["l_layer_type"][i]}' in layer {i}. Must be 0 (clay) or 1 (sand).")
            
        soil = _pile_and_soil.Soil(layers)
            
        prepared_forward_params = (pile, soil, d["P"], d["z_w"], d["N"], d["t_res_clay"])

        return prepared_forward_params

def create_scipy_ops(config):
    # Currently DOES NOT support variable pile diameter, due to pile_and_soil stuff

    def forward_model(*forward_params):
        res = _model_springs.solve_springs4(*prepare_for_scipy(*forward_params))

        # Check if any of "zeros" are nan, if so the result is probably wrong. (note, if all are nan it could mean P > P_ult which is expected behaviour)
        # "zeros" is the array of simultaneous equations results that should all be [almost] zero for a good solution.
        if np.any(np.isnan(res[4])) and not np.all(np.isnan(res[4])):
            raise RuntimeError("Some of the zeros are nan, some are not. This is unexpected.")
        
        # Check that if the results is all nan, then P > P_ult (otherwise it's an error)
        if np.all(np.isnan(res[4])):
            if res[8] / res[9] > 1:
                # print("P > P_ult! Nans will be returned")
                pass
            else:
                raise RuntimeError("All zeros are nan, but P <= P_ult. This is unexpected.")

        return res[0]

    def forward_log_likelihood(sigma, data, *forward_params):
        # Assuming additive gaussian white noise
        # If results are all jnp.nan, reflecting invalid parameter combination, set probability to zero
        res = forward_model(*forward_params)

        if np.all(np.isnan(res)):
            logp = -np.inf
        else:
            logp = (-np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((data - res) / sigma) ** 2).sum()

        return logp

    def logp_wrapper(sigma, data, *unordered_forward_params):
        return forward_log_likelihood(sigma, data, *_utilities.reorder_params(*unordered_forward_params, config=config))

    # Define a pytensor Op for our likelihood function
    class LogLikelihood(Op):
        # The inputs are parameters to be inferred. The other parameters are used from the global values.
        def make_node(self, data, *drawn_inferred_forward_params) -> Apply:
            # Convert inputs to tensor variables
            # Note the difference with black box op: pt.as_tensor_variable instead of pt.as_tensor
            inputs = [pt.as_tensor_variable(data)] + [pt.as_tensor_variable(x) for x in drawn_inferred_forward_params]

            # Define output type, in this case a single scalar
            outputs = [pt.scalar()]

            # Apply is an object that combines inputs, outputs and an Op (self)
            return Apply(self, inputs, outputs)

        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            # This is the method that compute numerical output given numerical inputs.

            log_likelihood = logp_wrapper(config.sigma, *inputs,
                                                 *config.fixed_forward_params_dict.values(),
                                                 *config.not_inferred_forward_params_dict.values())

            # Print out the log likelihood and input inferrable parameters
            if config.do_print:
                print(f"Printing log likelihood: {log_likelihood}")
                print(f"Printing sampled parameters")
                for i, inp in enumerate(inputs[1:]):
                    print(f"param {i}: {inp}")

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            outputs[0][0] = np.asarray(log_likelihood, dtype=node.outputs[0].dtype)

    logp_op = LogLikelihood()
    logp_grad_op = None # need jax for gradients

    def test_out(sigma, data, *drawn_inferred_forward_params):
        lik_out = logp_wrapper(sigma, data, *drawn_inferred_forward_params, *config.fixed_forward_params_dict.values(), *config.not_inferred_forward_params_dict.values())
        op_out = logp_op(data, *drawn_inferred_forward_params).eval()
        print(f"Forward out: {lik_out}, Op out: {op_out}")
        assert np.allclose(op_out, lik_out), f"Op output {op_out} does not match forward output {lik_out}"

    return forward_model, logp_op, logp_grad_op, test_out
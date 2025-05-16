import numpy as np
import jax
import jax.numpy as jnp

import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.gradient import grad_not_implemented

from . import _utilities
from . import _model_springs_jax

def create_jax_ops(config):

    def forward_model(*forward_params):
        res = _model_springs_jax.solve_springs_api_jax(*forward_params, throw=False)

        # Check if any of "zeros" are nan, if so the result is probably wrong.
        # "zeros" is the array of simultaneous equations results that should all be [almost] zero for a good solution.
        forces = jax.lax.cond(
            jnp.any(jnp.isnan(res[4])),
            lambda: jnp.full_like(res[0], jnp.nan),
            lambda: res[0], # forces
        )

        return forces

    def forward_log_likelihood(sigma, data, *forward_params):
        # Assuming additive gaussian white noise
        # If results are all jnp.nan, reflecting invalid parameter combination, set probability to zero
        res = forward_model(*forward_params)
        logp = jax.lax.cond(
            jnp.all(jnp.isnan(res)),
            lambda: -jnp.inf,
            lambda: (-jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((data - res) / sigma) ** 2).sum()
        )
        # concrete_logp = jax.lax.stop_gradient(logp)
        # concrete_gamma_d = jax.lax.stop_gradient(forward_params[grad_argnums[0]-2])
        # print(f"counter: {counter}, gamma_d: {concrete_gamma_d}, logp: {concrete_logp})")
        return logp

    jitted_forward = jax.jit(forward_model, static_argnums=config.forward_static_argnums)
    jitted_logp = jax.jit(forward_log_likelihood, static_argnums=config.likelihood_static_argnums)
    jitted_logp_grad = jax.jit(jax.grad(forward_log_likelihood,
                                                argnums=config.grad_argnums),
                                                static_argnums=config.likelihood_static_argnums)

    def jitted_logp_wrapper(sigma, data, *unordered_forward_params):
        return jitted_logp(sigma, data, *_utilities.reorder_params(*unordered_forward_params, config=config))

    def jitted_logp_grad_wrapper(sigma, data, *unordered_forward_params):
        return jitted_logp_grad(sigma, data, *_utilities.reorder_params(*unordered_forward_params, config=config))

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

            log_likelihood = jitted_logp_wrapper(config.sigma, *inputs,
                                                 *config.fixed_forward_params_dict.values(),
                                                 *config.not_inferred_forward_params_dict.values())

            # Print out the log likelihood and input inferrable parameters
            if config.do_print:
                concrete_logp = jax.lax.stop_gradient(log_likelihood)
                print(f"Printing log likelihood: {concrete_logp}")
                print(f"Printing sampled parameters")
                for i, inp in enumerate(inputs[1:]):
                    concrete_param = jax.lax.stop_gradient(inp)
                    print(f"param {i}: {concrete_param}")

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            outputs[0][0] = np.asarray(log_likelihood, dtype=node.outputs[0].dtype)

        def grad(self, inputs, output_gradients):
            out = logp_grad_op(*inputs)
            grads = out if isinstance(out, tuple) else (out,)

            print("printing grad, leninputs")
            print(grads)
            print(len(inputs))

            # If there are inputs for which the gradients will never be needed or cannot
            # be computed, `pytensor.gradient.grad_not_implemented` should  be used as the
            # output gradient for that input.
            output_gradient = output_gradients[0]

            # print("grad inputs: ", inputs, "grad: ", grads)
            return [grad_not_implemented(self, 0, inp) if i == 0 else
                output_gradient * grads[i-1] for i, inp in enumerate(inputs)]

    # Define a pytensor Op for the gradient of our likelihood function
    class LogLikelihoodGrad(Op):
        def make_node(self, data, *drawn_inferred_forward_params) -> Apply:
            # Convert inputs to tensor variables
            # Note the difference with black box op: pt.as_tensor_variable instead of pt.as_tensor
            inputs = [pt.as_tensor_variable(data)] + [pt.as_tensor_variable(x) for x in drawn_inferred_forward_params]

            # In practice, you should use
            # the exact dtype to avoid overhead when saving the results of the computation
            # in `perform`
            outputs = [x.type() for x in inputs[1:]]

            # Apply is an object that combines inputs, outputs and an Op (self)
            return Apply(self, inputs, outputs)

        def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
            grads = jitted_logp_grad_wrapper(config.sigma, *inputs,
                                             *config.fixed_forward_params_dict.values(),
                                             *config.not_inferred_forward_params_dict.values())

            # Print out the gradients
            if config.do_print:
                print(f"Printing gradients.")
                for i, g in enumerate(grads):
                    print(f"grad {i}: {g}")
                print()

            # Save the result in the outputs list provided by PyTensor
            # There is one list per output, each containing another list
            # pre-populated with a `None` where the result should be saved.
            for i, grad in enumerate(grads):
                outputs[i][0] = np.asarray(grad, dtype=node.outputs[i].dtype)

    # Create our Ops
    logp_op = LogLikelihood()
    logp_grad_op = LogLikelihoodGrad()

    # Create a function that verifies that the Ops work, by checking the Ops and jitted functions return the same thing
    def test_out(sigma, data, *drawn_inferred_forward_params):
        print("Compiling jit functions and testing out...")

        # Check that the jitted function and the Op return the same thing
        jitted_out = jitted_logp_wrapper(sigma, data, *drawn_inferred_forward_params, *config.fixed_forward_params_dict.values(), *config.not_inferred_forward_params_dict.values())
        op_out = logp_op(data, *drawn_inferred_forward_params).eval()
        print(f"Jitted out: {jitted_out}, Op out: {op_out}")
        assert np.allclose(op_out, jitted_out), f"Op output {op_out} does not match jitted output {jitted_out}"

        # Check that the gradient of the Op and the jitted function return the same thing
        jitted_grad = jitted_logp_grad_wrapper(sigma, data, *drawn_inferred_forward_params, *config.fixed_forward_params_dict.values(), *config.not_inferred_forward_params_dict.values())
        op_grad = logp_grad_op(data, *drawn_inferred_forward_params).eval()
        print(f"Jitted grad: {jitted_grad}, Op grad: {op_grad}")
        assert np.allclose(op_grad, jitted_grad), f"Op gradient {op_grad} does not match jitted gradient {jitted_grad}"

        print("Done compiling and testing out.")
    
    return jitted_forward, logp_op, logp_grad_op, test_out
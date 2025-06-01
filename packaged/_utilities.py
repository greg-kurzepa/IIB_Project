import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats

from . import _inference

# ---------------------------------------------------------------------------
# Utilities for PyMC Ops

def sort_x_by_y(x, y):
            return [X for (Y,X) in sorted(zip(y,x), key=lambda pair: pair[0])]

def reorder_params(*unordered_forward_params, config=None, unordered_argnames=None, ordered_argnames=None):
    if unordered_argnames is None:
        if config is None:
            raise ValueError("Either config or unordered_argnames must be provided.")
        unordered_argnames = config.wrapper_arg_order
    if ordered_argnames is None:
        ordered_argnames = _inference.forward_arg_order

    # Reorder unordered_forward_params in the same way that unordered_argnames would be reordered to form ordered_argnames
    unordered_idx_in_ordered = [ordered_argnames.index(x) for x in unordered_argnames]
    ordered_forward_params = sort_x_by_y(unordered_forward_params, unordered_idx_in_ordered)
    return ordered_forward_params

# ---------------------------------------------------------------------------
# Utilities for Plotting

def plot_2d_kde(x, y, ax=None, cmap=plt.cm.gist_earth_r, title="2D KDE Plot", xlabel=None, ylabel=None):

    do_show = False
    if ax is None:
        fig, ax = plt.subplots()
        do_show = True

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    values = np.vstack([x, y])
    kernel = scipy.stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.imshow(np.rot90(Z), cmap=cmap, extent=[xmin, xmax, ymin, ymax])
    ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)

    if do_show:
        fig.colorbar(label="Density")
        plt.show()
    else:
        return xmin, xmax, ymin, ymax

def plot_prior_samples(idata):

    for varname in list(idata.prior.data_vars):
        print(f"Plotting for samples from variable {varname}")
        prior_samples = np.array(idata.prior[varname])[0]

        assert prior_samples.shape[1] == 2, f"Expected prior samples to have shape (n_samples, 2), but got {prior_samples.shape}"

        # Get indexes of samples which had P > P_ult
        # i.e. the ones where the force profile is all np.nan
        over_limit_idxs = np.all(np.isnan(idata.prior_predictive["likelihood"]), axis=2)[0]

        over_limit_samples = prior_samples[over_limit_idxs]
        under_limit_samples = prior_samples[~over_limit_idxs]

        plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        plt.scatter(under_limit_samples[:, 0], under_limit_samples[:, 1], color='blue', label='$P \\leq P_\{ult\}$')
        plt.scatter(over_limit_samples[:, 0], over_limit_samples[:, 1], color='k', label='$P \\gt P_\{ult\}$')

        # KDE plot
        # _ = plot_2d_kde(under_limit_samples[:, 0], under_limit_samples[:, 1], ax=plt.gca(), title="KDE of Prior Samples")
        # plt.colorbar(label="Density")

        plt.show()
def sort_x_by_y(x, y):
            return [X for (Y,X) in sorted(zip(y,x), key=lambda pair: pair[0])]

def reorder_params(*unordered_forward_params, config=None, unordered_argnames=None, ordered_argnames=None):
    if unordered_argnames is None:
        if config is None:
            raise ValueError("Either config or unordered_argnames must be provided.")
        unordered_argnames = config.wrapper_arg_order
    if ordered_argnames is None:
        if config is None:
            raise ValueError("Either config or ordered_argnames must be provided.")
        ordered_argnames = config.forward_arg_order

    # Reorder unordered_forward_params in the same way that unordered_argnames would be reordered to form ordered_argnames
    unordered_idx_in_ordered = [ordered_argnames.index(x) for x in unordered_argnames]
    ordered_forward_params = sort_x_by_y(unordered_forward_params, unordered_idx_in_ordered)
    return ordered_forward_params
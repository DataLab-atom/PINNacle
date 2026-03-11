import torch
from src.optimizable.hard_constraint_optimizable import apply_ic_decay  # [OPTIMIZABLE]


def hard_constraint_wrapper(net, data, alpha=5):
    """
    Wrapper for hard constrain.
    output = ic + t * NN
    """

    def output_transform(inputs, outputs):
        t = inputs[..., -1:]
        ic_values = data.ic_func(inputs[..., :-1])
        return apply_ic_decay(t, outputs, ic_values, alpha)  # [OPTIMIZABLE]

    net._output_transform = output_transform
    return net

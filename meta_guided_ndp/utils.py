import numpy as np


def x0_sampling(dist, num_parameters):
    if dist == "U[0,1]":
        return np.random.rand(num_parameters)
    elif dist == "N(0,1)":
        return np.random.randn(num_parameters)
    elif dist == "U[-1,1]":
        return 2 * np.random.rand(num_parameters) - 1
    else:
        raise ValueError("Unknown distribution for x0")

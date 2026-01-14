import numpy as np

def sphere(x):
    """
    Sphere function (minimization problem)
    Global minimum at x = 0
    """
    return np.sum(x ** 2)


def rastrigin(x):
    """
    Rastrigin function (multimodal)
    Global minimum at x = 0
    """
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

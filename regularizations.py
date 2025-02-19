import numpy as np
import functions as fs

def L1(weight = 0):
    if not (isinstance(weight, (int, float)) or (isinstance(weight, np.ndarray) and weight.ndim == 0)):
        raise TypeError("Error: The weight for L1 regularization should be a single number")
    return lambda x: weight * fs.sum(fs.abs(x))

def L2(weight = 0):
    if not (isinstance(weight, (int, float)) or (isinstance(weight, np.ndarray) and weight.ndim == 0)):
        raise TypeError("Error: The weight for L2 regularization should be a single number")
    return lambda x: weight * fs.sum(x ** 2)
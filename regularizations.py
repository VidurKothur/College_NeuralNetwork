import numpy as np
from node import Node
import functions as fs

def L1(weight = 0):
    if not (isinstance(weight, (int, float)) or (isinstance(weight, np.ndarray) and weight.ndim == 0)):
        raise TypeError("Error: The weight for L1 regularization should be a single number")
    return lambda x: weight * fs.sum(fs.abs(x))
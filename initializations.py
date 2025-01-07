import numpy as np

def XavierNormal(dimension, xMean = 0, xConstant = 2):
    if not (isinstance(dimension, (list, tuple))):
        raise TypeError("Error: The dimensions for weight initialization must be a tuple or list")
    if not all(isinstance(dim, int) for dim in dimension):
        raise TypeError("Error: All values within the dimension should be a positive integer")
    if not (all(dim >= 1 for dim in dimension)):
        raise ValueError("Error: The dimensions for weight initialization should be all greater than zero")
    if not (isinstance(xConstant, (int, float)) or (isinstance(xConstant, np.ndarray) and xConstant.ndim == 0)):
        raise TypeError("Error: The xConstant for Xavier initialization should be a single number")
    if not (isinstance(xMean, (int, float)) or (isinstance(xMean, np.ndarray) and xConstant.ndim == 0)):
        raise TypeError("Error: The xMean for Xavier initialization should be a single number")
    denominator = np.sum(dimension, dtype=np.float64) if len(dimension) > 0 else 1e-8
    return lambda: np.random.normal(xMean, np.sqrt(xConstant / denominator), size = tuple(dimension)).tolist()
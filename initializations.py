import numpy as np
from data import Tensor

"""
This module contains initialization functions that can be used to set the initial values of parameters in the
neural network. Each function technically returns another function that is customized with the variable for the
means, standard deviations, and upper and lower bounds.

"""

def checkDimension(dimension):
    if not (isinstance(dimension, (list, tuple))):
        raise TypeError("Error: The dimensions for weight initialization must be a tuple or list")
    if not all(isinstance(dim, int) for dim in dimension):
        raise TypeError("Error: All values within the dimension should be a positive integer")
    if not (all(dim >= 1 for dim in dimension)):
        raise ValueError("Error: The dimensions for weight initialization should be all greater than zero")
    
def Constant(dimension, constant):
    checkDimension(dimension)
    if not (isinstance(constant, (int, float)) or (isinstance(constant, np.ndarray) and constant.ndim == 0)):
        raise TypeError("Error: The constant for Constant initialization must be a single number")
    return lambda: Tensor(constant * np.ones(dimension, dtype=np.float64))

def XavierNormal(dimension, xMean = 0, xConstant = 2):
    checkDimension(dimension)
    if not (isinstance(xConstant, (int, float)) or (isinstance(xConstant, np.ndarray) and xConstant.ndim == 0)):
        raise TypeError("Error: The xConstant for Xavier Normal initialization should be a single number")
    if not (isinstance(xMean, (int, float)) or (isinstance(xMean, np.ndarray) and xConstant.ndim == 0)):
        raise TypeError("Error: The xMean for Xavier Normal initialization should be a single number")
    denominator = np.sum(dimension, dtype=np.float64) if len(dimension) > 0 else 1e-8
    return lambda: Tensor(np.random.normal(xMean, np.sqrt(xConstant / denominator, dtype=np.float64), size = tuple(dimension), dtype=np.float64))

def XavierUniform(dimension, xConstant = 6):
    checkDimension(dimension)
    if not (isinstance(xConstant, (int, float)) or (isinstance(xConstant, np.ndarray) and xConstant.ndim == 0)):
        raise TypeError("Error: The xConstant for Xavier Uniform initialization should be a single number")
    bounds = np.sqrt(xConstant / np.sum(dimension), dtype=np.float64)
    return lambda: Tensor(np.random.uniform(-bounds, bounds, size = tuple(dimension), dtype=np.float64))

def HeNormal(dimension, hMean = 0, hConstant = 2):
    checkDimension(dimension)
    if not (isinstance(hConstant, (int, float)) or (isinstance(hConstant, np.ndarray) and hConstant.ndim == 0)):
        raise TypeError("Error: The hConstant for He Normal initialization should be a single number")
    if not (isinstance(hMean, (int, float)) or (isinstance(hMean, np.ndarray) and hMean.ndim == 0)):
        raise TypeError("Error: The hMean for He Normal initialization should be a single number")
    return lambda: Tensor(np.random.normal(hMean, np.sqrt(hConstant / (dimension[0] if len(dimension) > 0 else 1), dtype=np.float64), size = tuple(dimension), dtype=np.float64))

def HeUniform(dimension, hConstant = 2):
    checkDimension(dimension)
    if not (isinstance(hConstant, (int, float)) or (isinstance(hConstant, np.ndarray) and hConstant.ndim == 0)):
        raise TypeError("Error: The hConstant for He Uniform initialization should be a single number")
    bounds = np.sqrt(hConstant / (dimension[0] if len(dimension) > 0 else 1), dtype=np.float64)
    return lambda: Tensor(np.random.uniform(-bounds, bounds, size = tuple(dimension), dtype=np.float64))

def LecunNormal(dimension, lMean = 0, lConstant = 1):
    checkDimension(dimension)
    if not (isinstance(lConstant, (int, float)) or (isinstance(lConstant, np.ndarray) and lConstant.ndim == 0)):
        raise TypeError("Error: The lConstant for Lecun Normal initialization should be a single number")
    if not (isinstance(lMean, (int, float)) or (isinstance(lMean, np.ndarray) and lMean.ndim == 0)):
        raise TypeError("Error: The lMean for Lecun Normal initialization should be a single number")
    return lambda: Tensor(np.random.normal(lMean, np.sqrt(lConstant / (dimension[0] if len(dimension) > 0 else 1), dtype=np.float64), size = tuple(dimension), dtype=np.float64))

def LecunUniform(dimension, lConstant = 3):
    checkDimension(dimension)
    if not (isinstance(lConstant, (int, float)) or (isinstance(lConstant, np.ndarray) and lConstant.ndim == 0)):
        raise TypeError("Error: The lConstant for Lecun Uniform initialization should be a single number")
    bounds = np.sqrt(lConstant / (dimension[0] if len(dimension) > 0 else 1), dtype=np.float64)
    return lambda: Tensor(np.random.uniform(-bounds, bounds, size = tuple(dimension), dtype=np.float64))

def RandomNormal(dimension, rMean = 0, rConstant = 1):
    checkDimension(dimension)
    if not (isinstance(rConstant, (int, float)) or (isinstance(rConstant, np.ndarray) and rConstant.ndim == 0)):
        raise TypeError("Error: The rConstant for Random Normal initialization should be a single number")
    if not (isinstance(rMean, (int, float)) or (isinstance(rMean, np.ndarray) and rMean.ndim == 0)):
        raise TypeError("Error: The rMean for Random Normal initialization should be a single number")
    return lambda: Tensor(np.random.normal(rMean, rConstant, size = tuple(dimension), dtype=np.float64))

def RandomUniform(dimension, rConstant = 1):
    checkDimension(dimension)
    if not (isinstance(rConstant, (int, float)) or (isinstance(rConstant, np.ndarray) and rConstant.ndim == 0)):
        raise TypeError("Error: The rConstant for Random Uniform initialization should be a single number")
    return lambda: Tensor(np.random.uniform(-rConstant, rConstant, size = tuple(dimension), dtype=np.float64))
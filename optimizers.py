import numpy as np
from data import Tensor

"""
This module contains optimizers that update the value of parameters once their gradient has been found through
backpropagation.

Each optimizer returns a function that only takes an initial value and gradient, and the function is bound to
intermediate values that it would need to keep a running average, like the weightedAverage parameter in RMSProp
and the firstMoment and secondMoment parameters in Adam.

"""

def No():
    return lambda initial, gradient: initial

def SGD(learningRate):
    if not (isinstance(learningRate, (int, float)) or (isinstance(learningRate, np.ndarray) and learningRate.ndim == 0)):
        raise Exception("Error: The learning rate provided to the SGD optimizer must be a number")
    return lambda initial, gradient: initial - learningRate * gradient

def SGDMomentum(learningRate, decayConstant):
    if not (isinstance(learningRate, (int, float)) or (isinstance(learningRate, np.ndarray) and learningRate.ndim == 0)):
        raise Exception("Error: The learning rate provided to the SGDMomentum optimizer must be a number")
    if not ((isinstance(decayConstant, (int, float)) or (isinstance(decayConstant, np.ndarray) and decayConstant.ndim == 0)) and (decayConstant >= 0 and decayConstant <= 1)):
        raise Exception("Error: The decay constant provided to the SGDMomentum optimizer must be a number between 0 and 1")
    velocity = 0
    def inner(initial, gradient):
        nonlocal velocity
        final = initial - learningRate * velocity
        velocity = decayConstant * velocity + (1 - decayConstant) * initial
        return final
    return inner

def RMSProp(learningRate, rmsConstant):
    if not (isinstance(learningRate, (int, float)) or (isinstance(learningRate, np.ndarray) and learningRate.ndim == 0)):
        raise Exception("Error: The learning rate provided to the RMSProp optimizer must be a number")
    if not ((isinstance(rmsConstant, (int, float)) or (isinstance(rmsConstant, np.ndarray) and rmsConstant.ndim == 0)) and (rmsConstant >= 0 and rmsConstant <= 1)):
        raise Exception("Error: The decay constant provided to the RMSProp optimizer must be a number between 0 and 1")
    weightedAverage = 0
    def inner(initial, gradient):
        nonlocal weightedAverage
        weightedAverage = rmsConstant * weightedAverage + (1 - rmsConstant) * (gradient ** 2)
        return initial - (learningRate / Tensor(np.sqrt(weightedAverage + 1e-8))) * gradient
    return inner

def Adam(learningRate, velocityConstant, decayConstant):
    if not (isinstance(learningRate, (int, float)) or (isinstance(learningRate, np.ndarray) and learningRate.ndim == 0)):
        raise Exception("Error: The learning rate provided to the Adam optimizer must be a number")
    if not ((isinstance(velocityConstant, (int, float)) or (isinstance(velocityConstant, np.ndarray) and velocityConstant.ndim == 0)) and (velocityConstant >= 0 and velocityConstant <= 1)):
        raise Exception("Error: The velocity constant provided to the Adam optimizer must be a number between 0 and 1")
    if not ((isinstance(decayConstant, (int, float)) or (isinstance(decayConstant, np.ndarray) and decayConstant.ndim == 0)) and (decayConstant >= 0 and decayConstant <= 1)):
        raise Exception("Error: The decay constant provided to the Adam optimizer must be a number between 0 and 1")
    firstMoment = 0
    secondMoment = 0
    time = 0
    def inner(initial, gradient):
        nonlocal firstMoment, secondMoment, time
        firstMoment = velocityConstant * firstMoment + (1 - velocityConstant) * gradient
        secondMoment = decayConstant * secondMoment + (1 - secondMoment) * (gradient ** 2)
        final = initial - learningRate * ((firstMoment / (1 - velocityConstant ** time)) / (Tensor(np.sqrt(secondMoment / (1 - decayConstant ** time))) + 1e-8))
        time += 1
        return final
    return inner

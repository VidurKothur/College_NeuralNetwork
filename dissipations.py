from data import Tensor
from typing import *
import functions as fs

"""
This module provides some dissipation functions, which measure the difference between the value predicted by the neural
network and the actual value given in supervised learning. Each function runs the checkDimensions method to make sure that
the predicted and actual arrays are compatible with the operations which will be done on them.

This module uses functions from the functions module because this is still technically part of the training process, and
so any operation needs to travel through the computational graph, and therefore needs to travel through the functions
module.

"""

def checkDimensions(predicted: Any, actual: Any) -> Tensor:
    minLength = min(actual.ndim, predicted.value.data.ndim)
    if not (actual.shape[:minLength] == predicted.value.data.shape[:minLength]):
        raise Exception("Error: The actual and predicted arrays must be of the same shape")
    if actual.size == 0:
        raise Exception("Error: This loss function cannot be used on datasets that have zero entries")

def TotalSimpleError(actual, predicted) -> Tensor:
    checkDimensions(predicted, actual)
    return fs.sum(actual - predicted)

def MeanSimpleError(actual, predicted) -> Tensor:
    checkDimensions(predicted, actual)
    return fs.mean(actual - predicted)

def TotalSquaredError(predicted: Any, actual: Any) -> Tensor:
    checkDimensions(predicted, actual)
    return fs.sum((actual - predicted) ** 2)

def MeanSquaredError(actual, predicted) -> Tensor:
    checkDimensions(predicted, actual)
    return fs.mean((actual - predicted) ** 2)

def TotalAbsoluteError(predicted: Any, actual: Any) -> Tensor:
    checkDimensions(predicted, actual)
    return fs.sum(fs.abs(actual - predicted))

def MeanAbsoluteError(actual, predicted) -> Tensor:
    checkDimensions(predicted, actual)
    return fs.mean(fs.abs(actual - predicted))

def BinaryCrossEntropy(actual, predicted) -> Tensor:
    checkDimensions(predicted, actual)
    return -fs.mean(actual * fs.log(predicted) + (1 - actual) * fs.log(1 - predicted))

def CategoricalCrossEntropy(actual, predicted) -> Tensor:
    checkDimensions(predicted, actual)
    return -fs.mean(actual * fs.log(predicted))

def KLDivergence(actual, predicted) -> Tensor:
    checkDimensions(predicted, actual)
    return fs.sum(actual * fs.log(actual / predicted))

def HuberLoss(actual, predicted) -> Tensor:
    checkDimensions(predicted, actual)
    absErr = fs.abs(actual - predicted)
    return fs.mean(fs.where(absErr <= 1.0, 0.5 * absErr ** 2, absErr - 0.5))

def CosineSimilarity(actual, predicted) -> float:
    checkDimensions(predicted, actual)
    return fs.sum(actual * predicted) / (fs.linalg.norm(actual) * fs.linalg.norm(predicted))
import numpy as np
from typing import *
from node import Node
import functions as fs

def TotalSquaredError(predicted: Any, actual: Any) -> np.ndarray:
    if not (actual.shape == np.array(predicted.value).shape):
        raise Exception("Error: The actual and predicted arrays must be of the same shape")
    if actual.size == 0:
        raise Exception("Error: This loss function cannot be used on datasets that have zero entries")
    x2 = actual - predicted
    return fs.sum(x2 * x2)

def TotalAbsoluteError(predicted: Any, actual: Any) -> np.ndarray:
    minLength = min(actual.ndim, predicted.value.data.ndim)
    if not (actual.shape[:minLength] == predicted.value.data.shape[:minLength]):
        raise Exception("Error: The actual and predicted arrays must be of the same shape")
    if actual.size == 0:
        raise Exception("Error: This loss function cannot be used on datasets that have zero entries")
    x2 = actual - predicted
    return fs.sum(fs.abs(x2))
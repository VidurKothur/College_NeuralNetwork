import inspect
import numpy as np
from typing import *

class Variable:
    def __init__(self, debugForwardKey: str = None, debugBackwardKey: str = None, regularizationFunction: Callable = None, regularizationWeight: Union[int, float, np.ndarray] = 1, value: Any = None) -> None:
        self.debugForwardKey = None
        self.debugBackwardKey = None
        self.regularizationFunction = None
        self.regularizationWeight = None
        self.value = None
        if not debugForwardKey is None:
            if not (isinstance(debugForwardKey, str)):
                raise TypeError("Error: The provided forward debug key for parameters must be a string")
            self.debugForwardKey = debugForwardKey
        if not debugBackwardKey is None:
            if not (isinstance(debugBackwardKey, str)):
                raise TypeError("Error: The provided backward debug key for parameters must be a string")
            self.debugBackwardKey = debugBackwardKey
        if not regularizationFunction is None:
            if not (callable(regularizationFunction) and len(inspect.signature(regularizationFunction).parameters) == 1):
                raise ValueError("Error: The provided input 'regularization' should be a callable function that takes in one input, which is the weight")
            self.regularizationFunction = regularizationFunction
        if not (isinstance(regularizationWeight, (int, float)) or (isinstance(regularizationWeight, np.ndarray) and regularizationWeight.ndim == 0)):
            raise TypeError("Error: The regularization weight for this Variable should be a single number")
        self.regularizationWeight = regularizationWeight
        if value:
            try:
                self.value = np.array(value, dtype=np.float64)
            except:
                raise ValueError("Error: The provided input 'value' could not be resolved to a mathematical object")
            
    def setValue(self, value: Any) -> None:
        try:
            self.value = np.array(value, dtype=np.float64)
        except:
            raise ValueError("Error: The provided input 'value' could not be resolved to a mathematical object")

    def setRegularizationFunction(self, regularizationFunction: Callable = None, regularizationWeight: Union[int, float, np.ndarray] = None) -> None:
        if regularizationFunction:
            if not (callable(regularizationFunction) and len(inspect.signature(regularizationFunction).parameters) == 1):
                raise ValueError("Error: The provided input 'regularization' should be a callable function that takes in one input, which is the weight")
            self.regularizationFunction = regularizationFunction
        if not regularizationWeight is None:
            if not (isinstance(regularizationWeight, (int, float)) or (isinstance(regularizationWeight, np.ndarray) and regularizationWeight.ndim == 0)):
                raise TypeError("Error: The regularization weight for this Variable should be a single number")
            self.regularizationWeight = regularizationWeight

class Constant(Variable):
    def __init__(self, debugForwardKey: str = None, regularizationFunction: Callable = None, regularizationWeight: Union[int, float, np.ndarray] = 1, value = None):
        super().__init__(debugForwardKey, None, regularizationFunction, regularizationWeight, value)

class Parameter(Variable):
    def __init__(self, debugForwardKey: str = None, debugBackwardKey: str = None, regularizationFunction: Callable = None, regularizationWeight: Union[int, float, np.ndarray] = 1, initializationFunction: Callable = None, optimizationFunction: Callable = None, value = None):
        super().__init__(debugForwardKey, debugBackwardKey, regularizationFunction, regularizationWeight, value)
        self.initializationFunction = None
        self.optimizationFunction = None
        if not initializationFunction is None:
            if not (callable(initializationFunction) or len(inspect.signature(initializationFunction).parameters) == 0):
                raise ValueError("Error: The provided input 'initialization' should be a callable function that takes in no inputs")
            self.initializationFunction = initializationFunction
        if not optimizationFunction is None:
            if not (callable(optimizationFunction) or len(inspect.signature(optimizationFunction).parameters) == 2):
                raise ValueError("Error: The provided input 'optimizer' should be a callable function that takes in two inputs, a variable value and gradient value")
            self.optimizationFunction = optimizationFunction
        
    def setInitializationFunction(self, initializationFunction: Callable) -> None:
        if not (callable(initializationFunction) or len(inspect.signature(initializationFunction).parameters) == 0):
            raise ValueError("Error: The provided input 'initialization' should be a callable function that takes in no inputs")
        self.initializationFunction = initializationFunction
        
    def setOptimizationFunction(self, optimizationFunction: Callable) -> None:
        if not (callable(optimizationFunction) or len(inspect.signature(optimizationFunction).parameters) == 2):
            raise ValueError("Error: The provided input 'optimizer' should be a callable function that takes in two inputs, a variable value and gradient value")
        self.optimizationFunction = optimizationFunction
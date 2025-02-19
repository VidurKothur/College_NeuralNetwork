import inspect
from variables import *
from typing import *

"""
To be implemented later

"""

class Layer:
    def __init__(self, debugLayerKey: str = None, debugInputKey: str = None, constantList: list = [], parameterList: list = [], executionFunction: Callable = None) -> None:
        self.debugLayerKey = None
        self.debugInputKey = None
        self.constantList = []
        self.parameterList = []
        self.executionFunction = None
        self.initialized = False
        if not debugLayerKey is None:
            if not (isinstance(debugLayerKey, str)):
                raise TypeError("Error: The provided debug key for the layer must be a string")
            self.debugLayerKey = debugLayerKey
        if not debugInputKey is None:
            if not (isinstance(debugInputKey, str)):
                raise TypeError("Error: The provided debug key for the input must be a string")
            self.debugInputKey = debugInputKey
        if constantList != []:
            if not isinstance(constantList, list):
                raise TypeError("Error: The provided 'consts' input should be a list of Constants")
            if not all(isinstance(const, Constant) for const in constantList):
                raise TypeError("Error: The provided 'consts' input needs to have only Constant variables")
            self.constantList = constantList
        if parameterList != []:
            if not isinstance(parameterList, list):
                raise TypeError("Error: The provided 'learns' input should be a list of Constants")
            if not all(isinstance(learn, Constant) for learn in parameterList):
                raise TypeError("Error: The provided 'learns' input needs to have only Constant variables")
            self.parameterList = parameterList
        if not executionFunction is None:
            if not callable(executionFunction):
                raise TypeError("Error: The execution function for this layer is not a function")
            if not (len(inspect.signature(executionFunction).parameters.values()) == 3):
                raise ValueError("Error: The execution function should only take in three inputs, which are the input to the layer and the layer's constant and learned variables")
            self.executionFunction = executionFunction

    def addConstantVariable(self, variable: Constant) -> None:
        if not isinstance(variable, Constant):
            raise TypeError("Error: The provided 'variable' is not a Constant variable")
        self.constantList.append(variable)

    def addParameterVariable(self, variable: Parameter) -> None:
        if not isinstance(variable, Parameter):
            raise TypeError("Error: The provided 'variable' is not a Parameter variable")
        self.parameterList.append(variable)

    def initializeVariables(self) -> None:
        for const in self.constantList:
            if const.value is None:
                raise ValueError("Error: One of the Constant variables does not have a value, please provide a value and try again.")
        for param in self.parameterList:
            if param.value is None:
                if param.initializationFunction:
                    param.value = param.initializationFunction()
                else:
                    raise ValueError("Error: One of the Parameter variables does not have a value or initialization function, please provide either and try again.")
        self.initialized = True

    def setExecutionFunction(self, executionFunction: Callable) -> None:
        if not callable(executionFunction):
            raise TypeError("Error: The execution function for this layer is not a function")
        if not (len(inspect.signature(executionFunction).parameters.values()) == 3):
            raise ValueError("Error: The execution function should only take in three inputs, which are the input to the layer and the layer's constant and learned variables")
        self.executionFunction = executionFunction

    def execute(self, inputVector: Any):
        if self.executionFunction is None:
            raise ValueError("Error: The execution function for this layer doesn't exist, please define it before executing")
        if not self.initialized:
            raise ValueError("Error: The layer's variables have not been initialized, please initialize them before executing")
        try:
            if not self.debugLayerKey is None:
                print(f"{self.debugLayerKey}:\n----------------------------------------")
                print("Input:\n--------------------")
                if not self.debugInputKey is None:
                        print(f"{str(inputVector)}")
                print("--------------------")
                print("Constants:\n--------------------")
                for const in self.constantList:
                    if not const.debugForwardKey is None:
                        print(f"{const.debugForwardKey}: {str(const.value.value)}")
                print("--------------------")
                print("Learns:\n--------------------")
                for learn in self.parameterList:
                    if not learn.debugForwardKey is None:
                        print(f"{learn.debugForwardKey}: {str(learn.value.value)}")
                print("--------------------")
                print("----------------------------------------\n")
            inputVector = self.executionFunction(inputVector, self.constantList, self.parameterList)
            if inputVector is None:
                raise Exception("The execution function did not return a value to move to the next layer")
            return inputVector
        except Exception as e:
            raise Exception("Error: An error occured during the execution of this layer function:", str(e))
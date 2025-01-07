import numpy as np
from layers import *
import inspect
from gradient import gradient
from data import Tensor

class Network:
    def __init__(self):
        self.layers = []
        self.terminationFunction = None
        self.lossFunction = None
        self.dissipationFunction = None
        self.networkFunction = None
        self.dissipationDebugKey = None
        self.regularizationDebugKey = None
        self.lossDebugKey = None
                
    def addLayer(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(f"Error: The provided layer '{layer}' of type '{type(layer)}' is not a valid neural network layer")
        self.layers.append(layer)

    def setRegularizationFunction(self, regularizationFunction, regularizationWeight = 1):
        if not callable(regularizationFunction):
            raise TypeError(f"Error: The provided regularization function is not a function")
        if not len(inspect.signature(regularizationFunction).parameters.values()) == 1:
            raise ValueError(f"Error: The provided regularization function should take only one input, which is the weight")
        if not (isinstance(regularizationWeight, (int, float)) or (isinstance(regularizationWeight, np.ndarray) and regularizationWeight.ndim == 0)):
            raise TypeError("Error: The regularization weight for this Variable should be a single number")
        for layer in self.layers:
            for const in layer.constantList:
                if not const.regularizationFunction:
                    const.regularizationFunction = regularizationFunction
                if not const.regularizationWeight:
                    const.regularizationWeight = regularizationWeight
            for learn in layer.parameterList:
                if not learn.regularizationFunction:
                    learn.regularizationFunction = regularizationFunction
                if not learn.regularizationWeight:
                    learn.regularizationWeight = regularizationWeight

    def setOptimizationFunction(self, optimizationFunction):
        if not callable(optimizationFunction):
            raise TypeError(f"Error: The provided optimization function is not a function")
        if not len(inspect.signature(optimizationFunction).parameters.values()) == 2:
            raise ValueError(f"Error: The provided optimization function should take only two inputs, which are the previous input value and gradient value")
        for layer in self.layers:
            for learn in layer.parameterList:
                if not learn.optimizationFunction:
                    learn.optimizationFunction = optimizationFunction

    def setDissipationFunction(self, dissipationFunction):
        if not callable(dissipationFunction):
            raise TypeError(f"Error: The provided object for measuring network dissipation is not a function")
        if not len(inspect.signature(dissipationFunction).parameters.values()) == 2:
            raise ValueError(f"Error: The provided function for measuring network dissipation should take only two inputs, which are the network output and actual values")
        self.dissipationFunction = dissipationFunction

    def setLossFunction(self, lossFunction):
        if not callable(lossFunction):
            raise TypeError(f"Error: The provided function for combining regularization and dissipation loss is not a function")
        if not len(inspect.signature(lossFunction).parameters.values()) == 2:
            raise ValueError(f"Error: The provided function for combining regularization and dissipation loss should take only two inputs, which are the network loss and regularization values")
        
        def realLoss(dissipationDebugKey, regularizationDebugKey):
            def actualRealLoss(dissipation, regularization):
                if not dissipationDebugKey is None:
                    print(f"{dissipationDebugKey}: {dissipation.value}")
                if not regularizationDebugKey is None:
                    print(f"{regularizationDebugKey}: {regularization.value}")
                return lossFunction(dissipation, regularization)
            return actualRealLoss
        self.lossFunction = realLoss

    def debug(self, layerDebugKey = False, inputDebugKey = False, gradientDebugKey = False, constDebugKey = False, parameterDebugKey = False, dissipationDebugKey = False, regularizationDebugKey = False, lossDebugKey = False):
        if layerDebugKey:
            if not isinstance(layerDebugKey, str):
                raise TypeError("Error: The provided debug key for the layer must be a string")
            for layer in self.layers:
                layer.debugLayerKey = layerDebugKey
        if inputDebugKey:
            if not isinstance(inputDebugKey, str):
                raise TypeError("Error: The provided debug key for inputs must be a string")
            for layer in self.layers:
                layer.debugInputKey = inputDebugKey
        if gradientDebugKey:
            if not isinstance(gradientDebugKey, str):
                raise TypeError("Error: The provided debug key for gradients must be a string")
            for layer in self.layers:
                layer.debugGradientKey = gradientDebugKey
        if constDebugKey:
            if not isinstance(constDebugKey, str):
                raise TypeError("Error: The provided debug key for constants must be a string")
            for layer in self.layers:
                for const in layer.constantList:
                    const.debug = constDebugKey
        if parameterDebugKey:
            if not isinstance(parameterDebugKey, str):
                raise TypeError("Error: The provided debug key for parameters must be a string")
            for layer in self.layers:
                for learn in layer.parameterList:
                    learn.debug = parameterDebugKey
        if dissipationDebugKey:
            if not isinstance(dissipationDebugKey, str):
                raise TypeError("Error: The provided debug key for the network loss must be a string")
            self.dissipationDebugKey = dissipationDebugKey
        if regularizationDebugKey:
            if not isinstance(regularizationDebugKey, str):
                raise TypeError("Error: The provided debug key for the network loss must be a string")
            self.regularizationDebugKey = regularizationDebugKey
        if lossDebugKey:
            if not isinstance(lossDebugKey, str):
                raise TypeError("Error: The provided debug key for the network loss must be a string")
            self.lossDebugKey = lossDebugKey

    def setNetworkFunction(self, networkFunction, gradientOptions = {}):
        if not callable(networkFunction):
            raise TypeError(f"Error: The provided function for running the neural network is not a function")
        if not len(inspect.signature(networkFunction).parameters.values()) == 5:
            raise ValueError(f"Error: The provided function for running the neural network should take only five inputs, which are the data (input, ouptut), list of layer functions, dissipation function, regularization function, and loss function")
        if gradientOptions:
            if not isinstance(gradientOptions, dict):
                raise TypeError("Error: The gradientOptions parameter must be a dictionary which may or may not include the fields 'revert' and 'regulate'")

        @gradient(options = gradientOptions)
        def builder(data, layerFunctions, dissipationFunction, regularizationFunction, lossFunction):
            return networkFunction(data, layerFunctions, dissipationFunction, regularizationFunction, lossFunction)
        
        self.networkFunction = builder
        
    def train(self, data, epochs, batchSize):
        if not len(data) == 2:
            raise ValueError("Error: The data should have length 2, one list of inputs and one list of outputs")
        if not len(data[0]) == len(data[1]):
            raise ValueError("Error: The length of the inputs and outputs should be the same, each input should have an output")
        if not (isinstance(epochs, int) and epochs > 0):
            raise TypeError("Error: The epochs must be an integer greater than zero")
        if not (isinstance(batchSize, int) and batchSize > 0):
            raise TypeError("Error: The batch size must be an integer greater than zero")
        if not (len(data[0]) % batchSize == 0):
            raise ValueError("Error: The data should be able to split evenly into batches")
        try:
            inputs = np.array(data[0], np.float64)
            outputs = np.array(data[1], np.float64)
        except:
            raise ValueError("Error: The provided data could not be resolved to a mathematical object")
        layerFunctions = []
        for layer in self.layers:
            layerFunctions.append(layer.execute)
        for a in range(epochs):
            print(f"Epoch {a + 1}:\n--------------------------------------------------------------------------------\n")
            for b in range(len(data[0]) // batchSize):
                print(f"Round {b + 1}:\n------------------------------------------------------------\n")
                testInputs = Tensor(np.stack(inputs[b:b + batchSize]))
                testOutputs = Tensor(np.stack(outputs[b:b + batchSize]))
                try:
                    def regularizationFunction(layers):
                        def realRegFunc():
                            regularizationLoss = 0
                            for layer in layers:
                                for const in layer.constantList:
                                    if const.regularizationFunction:
                                        if const.value is None:
                                            raise Exception("A Constant variable's value was not defined")
                                        regularizationLoss += const.regularizationFunction(const.value)
                                for learn in layer.parameterList:
                                    if learn.regularizationFunction:
                                        if learn.value is None:
                                            raise Exception("A Parameter variable's value was not defined")
                                        regularizationLoss += learn.regularizationFunction(learn.value)
                            return regularizationLoss
                        return realRegFunc
                    self.networkFunction.buildFunction((testInputs, testOutputs), layerFunctions, self.dissipationFunction, regularizationFunction, self.lossFunction(self.dissipationDebugKey, self.regularizationDebugKey), self.layers)
                    loss = self.networkFunction.runFunction()
                    if not self.lossDebugKey is None:
                        print(f"{self.lossDebugKey}: {loss}")
                    self.networkFunction.runGradients()
                    self.networkFunction.destroyFunction(self.layers)
                except Exception as e:
                    raise Exception("Error: An error occured while running the neural network function:", str(e))
                print("------------------------------------------------------------\n")
            print("--------------------------------------------------------------------------------\n")
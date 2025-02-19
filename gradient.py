from functools import wraps
from inspect import signature
from graph import Graph
from node import Node

"""
This module provides a decorator that transforms regular functions into specialized functions, which run like normal
functions, but build up a computational graph and stores it within the _graph property of the function.

The bulidFunction method takes in the regular inputs of the function, along with a list of layers in the neural network
which does not get passed into the function, but the layers input is needed to convert the learnable parameters in each
layer into Node before being passed in, so that they can effectively contribute to the computational graph.

The runFunction method returns the value of the function run on the most recent set of inputs, so that the computational
graph does not need to get rebuilt when the function's previous output is desired.

The runGradient function returns the gradient of the function relative to the parameters that are learnable parameters,
and also automatically updates them through their optimizer functions.

The destroyFunction simply clears the computational graph node list so that a new computational graph is ready to be built.

"""

def gradient(options):
    def decorator(func):
        @wraps(func)
        def inner(*args):
            return func(*args)
        
        inner._graph = Graph()
        
        def buildFunction(data, layerFunctions, dissipationFunction, regularizationFunction, lossFunction, layers):
            count = 0
            counts = []
            for layer in layers:
                for const in layer.constantList:
                    const.value = Node(inner._graph, count, const.value, {count}, (), None, const.debugBackwardKey)
                    const.value.sendToGraph()
                    count += 1
                layerCounts = [layer.debugLayerKey, []]
                for learn in layer.parameterList:
                    layerCounts[1].append(count)
                    learn.value = Node(inner._graph, count, learn.value, {count}, (), None, learn.debugBackwardKey, optimizer=learn.optimizationFunction)
                    learn.value.sendToGraph()
                    inner._graph.gradients.append(count)
                    count += 1
                counts.append(layerCounts)
            final = func(data, layerFunctions, dissipationFunction, regularizationFunction(layers), lossFunction)
            inner._graph.result = final.identity
            inner._graph.built = True
            inner._graph.counts = counts

        def runFunction():
            return inner._graph.runFunction()

        def runGradient():
            inner._graph.runGradient(options)

        def destroyFunction(layers):
            for layer in layers:
                for const in layer.constantList:
                    const.value = const.value.value
                for learn in layer.parameterList:
                    learn.value = learn.value.value
            inner._graph.destroyFunction()

        inner.buildFunction = buildFunction
        inner.runFunction = runFunction
        inner.runGradients = runGradient
        inner.destroyFunction = destroyFunction

        return inner
    return decorator
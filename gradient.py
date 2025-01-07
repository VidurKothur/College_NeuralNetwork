from functools import wraps
from inspect import signature
from graph import Graph
from node import Node
from variables import Constant, Parameter
import numpy as np
from layers import Layer

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
            func(data, layerFunctions, dissipationFunction, regularizationFunction(layers), lossFunction)
            inner._graph.built = True
            inner._graph.counts = counts

        def runFunction():
            return inner._graph.registry[-1].value

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
import numpy as np

def SGD(learningRate):
    return lambda x, y: x - learningRate * y
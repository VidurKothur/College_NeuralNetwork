import numpy as np
from data import Tensor

def revertSum():
    def inner(oriShape, new):
        for a in range(len(new.shape) - len(oriShape)):
            new.data = np.sum(new.data, 0)
        for b in range(-1, -len(new.shape) - 1, -1):
            if oriShape[b] == 1 and new.shape[b] != 1:
                new.data = np.sum(new.data, b, keepdims=True)
            elif oriShape[b] != 1 and new.shape[b] == 1:
                new.data = np.repeat(new.data, oriShape[b], b)
        return new
    return inner

def revertMean():
    def inner(oriShape, new):
        for a in range(len(new.shape) - len(oriShape)):
            new.data = np.mean(new.data, 0)
        for b in range(-1, -len(new.shape) - 1, -1):
            if oriShape[b] == 1 and new.shape[b] != 1:
                new.data = np.mean(new.data, b, keepdims=True)
            elif oriShape[b] != 1 and new.shape[b] == 1:
                new.data = np.repeat(new.data, oriShape[b], b)
        return new
    return inner

def revertMax():
    def inner(oriShape, new):
        for a in range(len(new.shape) - len(oriShape)):
            new.data = np.max(new.data, 0)
        for b in range(-1, -len(new.shape) - 1, -1):
            if oriShape[b] == 1 and new.shape[b] != 1:
                new.data = np.max(new.data, b, keepdims=True)
            elif oriShape[b] != 1 and new.shape[b] == 1:
                new.data = np.repeat(new.data, oriShape[b], b)
        return new
    return inner

def revertMin():
    def inner(oriShape, new):
        for a in range(len(new.shape) - len(oriShape)):
            new.data = np.min(new.data, 0)
        for b in range(-1, -len(new.shape) - 1, -1):
            if oriShape[b] == 1 and new.shape[b] != 1:
                new.data = np.min(new.data, b, keepdims=True)
            elif oriShape[b] != 1 and new.shape[b] == 1:
                new.data = np.repeat(new.data, oriShape[b], b)
        return new
    return inner

def revertProd():
    def inner(oriShape, new):
        for a in range(len(new.shape) - len(oriShape)):
            new.data = np.prod(new.data, 0)
        for b in range(-1, -len(new.shape) - 1, -1):
            if oriShape[b] == 1 and new.shape[b] != 1:
                new.data = np.prod(new.data, b, keepdims=True)
            elif oriShape[b] != 1 and new.shape[b] == 1:
                new.data = np.repeat(new.data, oriShape[b], b)
        return new
    return inner

def gradientClip():
    def inner(gradient):
        return Tensor(np.clip(gradient.data, np.finfo(np.float64).min, np.finfo(np.float64).max))
    return inner

def normalizeL1():
    def inner(gradient):
        norm = np.sqrt(np.sum(gradient.data ** 2))
        return Tensor(gradient.data / (norm + 1e-8))
    return inner

def normalizeL2():
    def inner(gradient):
        norm = np.sum(np.abs(gradient.data))
        return Tensor(gradient / (norm + 1e-8))
    return inner

def scale(factor):
    try:
        factor = np.array(factor)
    except:
        raise TypeError(f"Error: '{factor}' (type {type(factor)}) could not be resolved to a mathematical object")
    def inner(gradient):
        try:
            return Tensor(gradient.data * factor)
        except Exception as e:
            raise ValueError(f"Error: '{gradient}' could not be scaled by '{factor}'. Reason: {str(e)}")
    return inner
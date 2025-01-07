import numpy as np
from node import Node
import options as ops
from typing import *
from data import Tensor
import copy

def toArray(a: Any) -> np.ndarray[np.float64]:
    try:
        if isinstance(a, Tensor):
            return a.data
        return np.array(a, np.float64)
    except:
        raise TypeError(f"Error: '{a}' (type {type(a)}) could not be resolved to a mathematical object")
    
def eqDim(x, y):
    if x.ndim < y.ndim:
        while x.ndim < y.ndim:
            x = np.array([x])
        return x, y
    elif x.ndim > y.ndim:
        while x.ndim > y.ndim:
            y = np.array([y])
        return x, y
    else:
        return x, y
    
def broadcast(a, s):
    if isinstance(a, Node):
        try:
            x = copy.deepcopy(np.broadcast_to(a.value.data, s))
        except:
            raise ValueError(f"Error: The operand {a.value.data} and could not be broadcasted to {s}")
        if not x.shape == a.value.shape:
            identity = len(a.graph.registry)
            s1 = Node(a.graph, identity, Tensor(s), {identity}, (), None, False, None)
            s1.sendToGraph()
            x1 = Node(a.graph, identity + 1, Tensor(x), a.paths | {identity, identity + 1}, (a.identity, identity), broadGrad, False, None)
            x1.sendToGraph()
            x = x1
        else:
            x = a
        return x
    else:
        try:
            try:
                a = Tensor(a)
            except: 
                raise TypeError(f"Error: The operand {a} could not be resolved to a mathematical object")
            x = np.broadcast_to(a.data, s)
        except:
            raise ValueError(f"Error: The operand {a} could not be broadcasted to {s}")
        return Tensor(x)

def func2(a: Any, b: Any, op: Callable, fun: Callable, prompt: str, broad: bool = True) -> Union[Node, np.ndarray[np.float64]]:
    if isinstance(a, Node) and isinstance(b, Node):
        a.value = Tensor(a.value)
        b.value = Tensor(b.value)
        if broad:
            shape = np.broadcast_shapes(a.value.shape, b.value.shape)
            a = broadcast(a, shape)
            b = broadcast(b, shape)
        try:
            v = op(a.value, b.value)
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{a.value}' and '{b.value}'. Reason: {str(e)}")
        identity = len(a.graph.registry)
        c = Node(a.graph, identity, v, a.paths | b.paths | {identity}, (a.identity, b.identity), fun, False, None)
        c.sendToGraph()
        return c
    elif isinstance(a, Node) and not isinstance(b, Node):
        a.value = Tensor(a.value)
        b = Tensor(b)
        identity = len(a.graph.registry)
        c = Node(a.graph, identity, b, {identity}, (), None, False, None)
        c.sendToGraph()
        if broad:
            shape = np.broadcast_shapes(a.value.shape, c.value.shape)
            a = broadcast(a, shape)
            c = broadcast(c, shape)
        try:
            v = op(a.value, b)
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{a.value}' and '{b}'. Reason: {str(e)}")
        d = Node(a.graph, identity + 1, v, a.paths | c.paths | {identity + 1}, (a.identity, c.identity), fun, False, None)
        d.sendToGraph()
        return d
    elif not isinstance(a, Node) and isinstance(b, Node):
        a = Tensor(a)
        b.value = Tensor(b.value)
        identity = len(b.graph.registry)
        c = Node(b.graph, identity, a, {identity}, (), None, False, None)
        c.sendToGraph()
        if broad:
            shape = np.broadcast_shapes(c.value.shape, b.value.shape)
            c = broadcast(c, shape)
            b = broadcast(b, shape)
        try:
            v = op(a, b.value)
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{a}' and '{b.value}'. Reason: {str(e)}")
        d = Node(b.graph, identity + 1, v, b.paths | c.paths | {identity + 1}, (c.identity, b.identity), fun, False, None)
        d.sendToGraph()
        return d
    else:
        a = Tensor(a)
        b = Tensor(b)
        if broad:
            shape = np.broadcast_shapes(a.shape, b.shape)
            a = broadcast(a, shape)
            b = broadcast(b, shape)
        try:
            c = op(a, b)
            return c
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{a}' and '{b}'. Reason: {str(e)}")
        
def func1(a: Any, op: Callable, fun: Callable, prompt: str) -> Union[Node, np.ndarray[np.float64]]:
    if isinstance(a, Node):
        a.value = Tensor(a.value)
        try:
            v = op(a.value)
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{a.value}'. Reason: {str(e)}")
        identity = len(a.graph.registry)
        c = Node(a.graph, identity, v, a.paths | {identity}, (a.identity,), fun, False, None)
        c.sendToGraph()
        return c
    else:
        a = Tensor(a)
        try:
            c = op(a)
            return c
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{a}'. Reason: {str(e)}")
        
def grad2(a: Any, b: Any, aFunc: Callable, bFunc: Callable, pos: int, options: dict, prompt: str) -> np.ndarray:
    a = Tensor(a)
    b = Tensor(b)
    final = None
    try:
        if pos == 0:
            final = aFunc(a, b)
        else:
            final = bFunc(a, b)
    except Exception as e:
        raise ValueError(f"Error: {prompt} gradient could not be calculated on '{a}' and '{b}'. Reason: {str(e)}")
    if "revert" in options:
        if pos == 0:
            final = options["revert"](a.shape, final)
        else:
            final = options["revert"](b.shape, final)
    if "regulate" in options:
        final = options["regulate"](final.data)
    return final

def grad1(a: Any, aFunc: Callable, pos: int, options: dict, prompt: str) -> np.ndarray:
    a = Tensor(a)
    final = None
    try:
        if pos == 0:
            final = aFunc(a)
    except Exception as e:
        raise ValueError(f"Error: {prompt} gradient could not be calculated on '{a}'. Reason: {str(e)}")
    if "revert" in options:
        if pos == 0:
            final = options["revert"](a.shape, final)
    if "regulate" in options:
        final = options["regulate"](final)
    return final

def add(a: Any, b: Any) -> np.ndarray:
    op = lambda x, y: Tensor(x.data + y.data)
    return func2(a, b, op, addGrad, "Addition")

def addGrad(a: Any, b: Any, pos: int, options: dict = { "revert": ops.revertSum(), "regulate": None }):
    aF = lambda x, y: Tensor(np.ones_like(y.data, np.float64))
    bF = lambda x, y: Tensor(np.ones_like(x.data, np.float64))
    return grad2(a, b, aF, bF, pos, options, "Addition")

def sub(a: Any, b: Any) -> np.ndarray:
    op = lambda x, y: Tensor(x.data - y.data)
    return func2(a, b, op, subGrad, "Subtraction")

def subGrad(a: Any, b: Any, pos: int, options: dict = { "revert": ops.revertSum(), "regulate": None }):
    aF = lambda x, y: Tensor(np.ones_like(y.data, np.float64))
    bF = lambda x, y: Tensor(-np.ones_like(x.data, np.float64))
    return grad2(a, b, aF, bF, pos, options, "Subtraction")

def mul(a, b):
    op = lambda x, y: Tensor(x.data * y.data)
    return func2(a, b, op, mulGrad, "Multiplication")

def mulGrad(a, b, pos, options = { "revert": ops.revertSum(), "regulate": None }):
    aF = lambda x, y: Tensor(y.data)
    bF = lambda x, y: Tensor(x.data)
    return grad2(a, b, aF, bF, pos, options, "Multiplication")

def mmul(a, b):
    def op(x, y):
        x.data, y.data = eqDim(x.data, y.data)
        return Tensor(x.data @ y.data)
    return func2(a, b, op, mmulGrad, "Matrix Multiplication", False)

def mmulGrad(a, b, pos, options = { "revert": ops.revertSum(), "regulate": None }):
    aF = lambda x, y: Tensor(y.data.T)
    bF = lambda x, y: Tensor(x.data.T)
    return grad2(a, b, aF, bF, pos, options, "Matrix Multiplication")

def sum(a):
    op = lambda x: Tensor(np.sum(x.data, dtype=np.float64))
    return func1(a, op, sumGrad, "Summation")

def sumGrad(a, pos, options = { "revert": ops.revertSum(), "regulate": None }):
    aF = lambda x: Tensor(np.ones_like(x.data, dtype=np.float64))
    return grad1(a, aF, pos, options, "Summation")

def abs(a):
    op = lambda x: Tensor(np.abs(x.data, dtype=np.float64))
    return func1(a, op, absGrad, "Absolute Value")

def absGrad(a, pos, options = { "revert": ops.revertSum(), "regulate": None }):
    aF = lambda x: Tensor(np.sign(x.data, dtype=np.float64))
    return grad1(a, aF, pos, options, "Absolute Value")

def ReLU(a):
    op = lambda x: Tensor(np.where(x.data < 0, 0, x.data))
    return func1(a, op, ReLUGrad, "Rectified Linear Unit")

def ReLUGrad(a, pos, options = { "revert": ops.revertSum(), "regulate": None }):
    aF = lambda x: Tensor(np.where(x.data < 0, 0, 1))
    return grad1(a, aF, pos, options, "Rectified Linear Unit")

def broadGrad(a, pos, options = { "revert": ops.revertSum(), "regulate": None }):
    def aF(x, s):
        x1 = np.broadcast_to(x.data, s)
        s1 = x.shape
        axes = tuple(i for i, (xdim, sdim) in enumerate(zip(tuple(1 for j in range(len(s) - len(s1))) + s1, s)) if xdim == 1 and sdim > 1)
        return Tensor(np.sum(x1, axis=axes, keepdims=True).reshape(s1))
    return grad1(a, aF, pos, options, "Broadcast")
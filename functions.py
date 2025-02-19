import numpy as np
from node import Node
from typing import *
from data import Tensor

e = 2.718281828459045
pi = 3.141592653589793

"""
Below are the wrapping functions for handling operations during forward propagation.

func2 is a wrapping function for handling a function of two inputs.

If both inputs are of type Node, then func2 will:
1. Make sure both input Nodes are wrapping data that is of type Tensor
2. Broadcast or map the inputs if they need to be broadcasted or mapped to the same dimension
3. Run the operation on the Tensor data that is inside the node
4. Create a resulting Node, send that to the Graph, and return the Node

If one input is of type Node and another is not, then func2 will:
1. Make sure that both the Node's internal data and the other input are of type Tensor
2. Wrap the other input inside a Node and send that to the Graph
3. Broadcast or map the inputs if they need to be broadcasted or mapped to the same dimension
4. Run the operation on the Tensor data that is inside the node
5. Create a resulting Node, send that to the Graph, and return the Node

If both inputs are not of type Node, then func2 will:
1. Make sure both inputs are of type Tensor
2. Broadcast or map the inputs if they need to be broadcasted or mapped to the same dimension
3. Run the operation on the Tensors and return the result

NOTE: The operation function will only take in two Tensors as input and will only output a single Tensor.

func1 is a wrapping function for handling a function of one input.

If the input is of type Node, then func1 will:
1. Make sure the Node's internal data is of type Tensor
2. Run the operation on the Tensor data
3. Create a resulting Node, send that to the Graph, and return the Node

If the input is not of type Node, then func1 will:
1. Make sure the input is of type Tensor
2. Run the operation on the Tensor and return it

NOTE: The operation function will only take in a single Tensor as input and will only output a single Tensor.
NOTE: Broadcasting and mapping will not occur in func1 because there will only be one input.

"""

def func2(input1: Any, input2: Any, operation: Callable, gradient: Callable, prompt: str, matchDimension: bool = True, toBroadcast: bool = True) -> Union[Node, Tensor]:
    if isinstance(input1, Node) and isinstance(input2, Node):
        input1.value = Tensor(input1.value)
        input2.value = Tensor(input2.value)
        if toBroadcast:
            shape = np.broadcast_shapes(input1.value.shape, input2.value.shape)
            input1 = broadcast(input1, shape)
            input2 = broadcast(input2, shape)
        if matchDimension:
            dim1 = input1.value.ndim
            dim2 = input2.value.ndim
            if dim1 < dim2:
                input1 = dimension(input1, dim2 - dim1)
            elif dim2 < dim1:
                input2 = dimension(input2, dim1 - dim2)
        try:
            value = operation(input1.value, input2.value)
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{input1.value}' and '{input2.value}'. Reason: {str(e)}")
        identity = len(input1.graph.registry)
        result = Node(input1.graph, identity, value, input1.paths | input2.paths | {identity}, (input1.identity, input2.identity), gradient, False, None)
        result.sendToGraph()
        return result
    elif isinstance(input1, Node) and not isinstance(input2, Node):
        input1.value = Tensor(input1.value)
        input2 = Tensor(input2)
        identity = len(input1.graph.registry)
        operand = Node(input1.graph, identity, input2, {identity}, (), None, False, None)
        operand.sendToGraph()
        if toBroadcast:
            shape = np.broadcast_shapes(input1.value.shape, operand.value.shape)
            input1 = broadcast(input1, shape)
            operand = broadcast(operand, shape)
        if matchDimension:
            dim1 = input1.value.ndim
            dim2 = operand.value.ndim
            if dim1 < dim2:
                input1 = dimension(input1, dim2 - dim1)
            if dim2 < dim1:
                operand = dimension(operand, dim1 - dim2)
        try:
            value = operation(input1.value, operand.value)
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{input1.value}' and '{operand.value}'. Reason: {str(e)}")
        result = Node(input1.graph, len(input1.graph.registry), value, input1.paths | operand.paths | {identity + 1}, (input1.identity, operand.identity), gradient, False, None)
        result.sendToGraph()
        return result
    elif not isinstance(input1, Node) and isinstance(input2, Node):
        input1 = Tensor(input1)
        input2.value = Tensor(input2.value)
        identity = len(input2.graph.registry)
        operand = Node(input2.graph, identity, input1, {identity}, (), None, False, None)
        operand.sendToGraph()
        if toBroadcast:
            shape = np.broadcast_shapes(operand.value.shape, input2.value.shape)
            operand = broadcast(operand, shape)
            input2 = broadcast(input2, shape)
        if matchDimension:
            dim1 = operand.value.ndim
            dim2 = input2.value.ndim
            if dim1 < dim2:
                operand = dimension(operand, dim2 - dim1)
            if dim2 < dim1:
                input2 = dimension(input2, dim1 - dim2)
        try:
            value = operation(operand.value, input2.value)
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{operand.value}' and '{input2.value}'. Reason: {str(e)}")
        result = Node(input2.graph, len(input2.graph.registry), value, operand.paths | input2.paths | {identity + 1}, (operand.identity, input2.identity), gradient, False, None)
        result.sendToGraph()
        return result
    else:
        input1 = Tensor(input1)
        input2 = Tensor(input2)
        if toBroadcast:
            shape = np.broadcast_shapes(input1.shape, input2.shape)
            input1 = broadcast(input1, shape)
            input2 = broadcast(input2, shape)
        if matchDimension:
            dim1 = input1.ndim
            dim2 = input2.ndim
            if dim1 < dim2:
                input1 = dimension(input1, dim2 - dim1)
            if dim2 < dim1:
                input2 = dimension(input2, dim1 - dim2)
        try:
            result = operation(input1, input2)
            return result
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{input1}' and '{input2}'. Reason: {str(e)}")
        
def func1(input1: Any, operation: Callable, gradient: Callable, prompt: str) -> Union[Node, np.ndarray[np.float64]]:
    if isinstance(input1, Node):
        input1.value = Tensor(input1.value)
        try:
            value = operation(input1.value)
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{input1.value}'. Reason: {str(e)}")
        identity = len(input1.graph.registry)
        result = Node(input1.graph, identity, value, input1.paths | {identity}, (input1.identity,), gradient, False, None)
        result.sendToGraph()
        return result
    else:
        input1 = Tensor(input1)
        try:
            result = operation(input1)
            return result
        except Exception as e:
            raise ValueError(f"Error: {prompt} could not be performed on '{input1}'. Reason: {str(e)}")

"""
Below are the functions for broadcasting two arrays or making their dimensions equal, along with their
respective gradient functions.

Both functions add to the respective computational graph if the input is a Node, but the functions and
gradients will not pass through func1/func2 and grad1/grad2 because these two functions are exceptions, 
there is no need to write code for finding a gradient with respect to the shape or number of dimensions
to expand.

"""

def dimension(input1: Union[Node, Tensor], number: int) -> Union[Node, Tensor]:
    if isinstance(input1, Node):
        final = None
        if number != 0:
            try:
                final = np.expand_dims(input1.value.data, axis=tuple(a for a in range(0, number))).copy()
            except:
                raise ValueError(f"Error: The operand {input1.value.data} could not expanded {number} dimensions")
            identity = len(input1.graph.registry)
            numberNode = Node(input1.graph, identity, Tensor(number), {identity}, (), None, False, None)
            numberNode.sendToGraph()
            newFinal = Node(input1.graph, identity + 1, Tensor(final), input1.paths | {identity, identity + 1}, (input1.identity, identity), dimensionGrad, False, None)
            newFinal.sendToGraph()
            final = newFinal
        else:
            final = input1
        return final
    else:
        try:
            final = np.expand_dims(input1.data, axis=tuple(a for a in range(0, number))).copy()
            return Tensor(final)
        except:
            raise ValueError(f"Error: The operand {input1} could not expanded {number} dimensions")

def dimensionGrad(input1: Tensor, position: int):
    return input1

def broadcast(input1: Union[Node, Tensor], shape: tuple) -> Union[Node, Tensor]:
    if isinstance(input1, Node):
        final = None
        if input1.value.shape != shape:
            try:
                final = np.broadcast_to(input1.value.data, shape).copy()
            except:
                raise ValueError(f"Error: The operand {input1.value.data} and could not be broadcasted to {shape}")
            identity = len(input1.graph.registry)
            shapeNode = Node(input1.graph, identity, Tensor(shape), {identity}, (), None, False, None)
            shapeNode.sendToGraph()
            newFinal = Node(input1.graph, identity + 1, Tensor(final), input1.paths | {identity, identity + 1}, (input1.identity, identity), broadcastGrad, False, None)
            newFinal.sendToGraph()
            final = newFinal
        else:
            final = input1
        return final
    else:
        try:
            final = np.broadcast_to(input1.data, shape)
            return Tensor(final)
        except:
            raise ValueError(f"Error: The operand {input1} could not be broadcasted to {shape}")
    
def broadcastGrad(input1: Tensor, shape: Tensor, position: int):
    output = np.broadcast_to(input1.data, shape).copy()
    originalShape = input1.shape
    axes = tuple(i for i, (xDim, shapeDim) in enumerate(zip(tuple(1 for j in range(len(shape) - len(originalShape))) + originalShape, shape)) if xDim == 1 and shapeDim > 1)
    return Tensor(np.sum(output, axis=axes, keepdims=True).reshape(originalShape))

"""
Below are the external-use functions that this module supports, along with their respective gradient functions.

Each function will have this format:

Binary Operation:

def name(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor({some operation of inputs})
    return func2(input1=input1, input2=input2, operation=operation, gradient={respective gradient}, 
    prompt="{function name}", dimension=False, broadcast=True)

Unary Operation:

def name(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor({some operation of input})
    return func1(input1=input1, operation=operation, gradient={respective gradient}, prompt="{function name}")

In both cases, the inputs will be rerouted into the wrapper function (func1 or func2), where type checking, 
computational graph handling, and error handling will occur. The "prompt" parameter is only used for 
customizing the error message, and the "gradient" parameter is only sent for Nodes to carry during backpropagation.

NOTE: The operation will always take in Tensors as input and output Tensors.

Each gradient will have this format:

Binary Gradient:

def nameGrad(input1: Tensor, input2: Tensor, position: int):
    if position == 0:
        return Tensor({some operation of inputs})
    else:
        return Tensor({some operation of inputs})

Unary Gradient:

def nameGrad(input1: Tensor, position: int):
    return Tensor({some operation of inputs})

In both cases, the return value will be handled differently, depending on what the gradient is being taken
respect to, the "position" parameter. There is no error handling here because the data types are guaranteed
to be Tensor by the backpropagation algorithm, and the error will be handled separately during network
execution.

"""

def add(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(x.data + y.data)
    return func2(input1=input1, input2=input2, operation=operation, gradient=addGrad, prompt="Addition", matchDimension=False, toBroadcast=True)

def addGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative * Tensor(np.ones_like(input2.data, np.float64)) if not localDerivative is None else Tensor(np.ones_like(input2.data, np.float64))
    else:
        return localDerivative * Tensor(np.ones_like(input1.data, np.float64)) if not localDerivative is None else Tensor(np.ones_like(input1.data, np.float64))

def sub(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(x.data - y.data)
    return func2(input1=input1, input2=input2, operation=operation, gradient=subGrad, prompt="Subtraction", matchDimension=False, toBroadcast=True)

def subGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative * Tensor(np.ones_like(input2.data, np.float64)) if not localDerivative is None else Tensor(np.ones_like(input2.data, np.float64))
    else:
        return localDerivative * Tensor(-np.ones_like(input1.data, np.float64)) if not localDerivative is None else Tensor(-np.ones_like(input2.data, np.float64))

def mul(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(x.data * y.data)
    return func2(input1=input1, input2=input2, operation=operation, gradient=mulGrad, prompt="Multiplication", matchDimension=False, toBroadcast=True)

def mulGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative * Tensor(input2.data) if not localDerivative is None else Tensor(input2.data)
    else:
        return localDerivative * Tensor(input1.data) if not localDerivative is None else Tensor(input1.data)

def div(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(x.data / y.data)
    return func2(input1=input1, input2=input2, operation=operation, gradient=divGrad, prompt="Division", matchDimension=False, toBroadcast=True)

def divGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative * Tensor(1 / input2.data) if not localDerivative is None else Tensor(1 / input2.data)
    else:
        return localDerivative * Tensor(-input1.data / (input2.data ** 2)) if not localDerivative is None else Tensor(-input1.data / (input2.data ** 2))

def pow(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(x.data ** y.data)
    return func2(input1=input1, input2=input2, operation=operation, gradient=powGrad, prompt="Power", matchDimension=False, toBroadcast=True)

def powGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative * Tensor((input2.data) * (input1.data ** (input2.data - 1))) if not localDerivative is None else Tensor((input2.data) * (input1.data ** (input2.data - 1)))
    else:
        return localDerivative * Tensor((input1.data ** input2.data) * np.log(input1.data, dtype=np.float64)) if not localDerivative is None else Tensor((input1.data ** input2.data) * np.log(input1.data, dtype=np.float64))

def log(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(np.log(x.data, dtype=np.float64) / np.log(y.data, dtype=np.float64))
    return func2(input1=input1, input2=input2, operation=operation, gradient=logGrad, prompt="Logarithm", matchDimension=False, toBroadcast=True)

def logGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative * Tensor(-np.log(input2.data, dtype=np.float64) / (input1.data * (np.log(input1.data, dtype=np.float64) ** 2))) if not localDerivative is None else Tensor(-np.log(input2.data, dtype=np.float64) / (input1.data * (np.log(input1.data, dtype=np.float64) ** 2)))
    else:
        return localDerivative * Tensor(1 / (input2.data * np.log(input1.data, dtype=np.float64))) if not localDerivative is None else Tensor(1 / (input2.data * np.log(input1.data, dtype=np.float64)))
    
def mmul(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(x.data @ y.data)
    return func2(input1=input1, input2=input2, operation=operation, gradient=mmulGrad, prompt="Matrix Multiplication", matchDimension=True, toBroadcast=False)

def mmulGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative @ Tensor(input2.T) if not localDerivative is None else Tensor(input2.T @ np.ones(np.shape(input1 @ input2), np.float64))
    else:
        return Tensor(input1.T) @ localDerivative if not localDerivative is None else Tensor(input1.T @ np.ones(np.shape(input1 @ input2), np.float64))

def max(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(np.maximum(x.data, y.data, dtype=np.float64))
    return func2(input1=input1, input2=input2, operation=operation, gradient=maxGrad, prompt="Maximum", matchDimension=True, toBroadcast=False)

def maxGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative * Tensor((input1.data >= input2.data).float()) if not localDerivative is None else Tensor((input1.data >= input2.data).float())
    else:
        return localDerivative * Tensor((input1.data <= input2.data).float()) if not localDerivative is None else Tensor((input1.data <= input2.data).float())

def min(input1: Any, input2: Any) -> Union[Node, Tensor]:
    operation = lambda x, y: Tensor(np.minimum(x.data, y.data))
    return func2(input1=input1, input2=input2, operation=operation, gradient=minGrad, prompt="Minimum", matchDimension=True, toBroadcast=False)

def minGrad(input1: Tensor, input2: Tensor, position: int, localDerivative: Tensor = None):
    if position == 0:
        return localDerivative * Tensor((input1.data <= input2.data).float()) if not localDerivative is None else Tensor((input1.data <= input2.data).float())
    else:
        return localDerivative * Tensor((input1.data >= input2.data).float()) if not localDerivative is None else Tensor((input1.data >= input2.data).float())

def exp(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.exp(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=expGrad, prompt="E to the x")

def expGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(np.exp(input1.data, dtype=np.float64)) if not localDerivative is None else Tensor(np.exp(input1.data, dtype=np.float64))

def ln(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.log(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=lnGrad, prompt="Natural Logarithm")

def lnGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(1 / (input1.data)) if not localDerivative is None else Tensor(1 / (input1.data))

def sin(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.sin(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=sinGrad, prompt="Sine")

def sinGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(np.cos(input1.data, dtype=np.float64)) if not localDerivative is None else Tensor(np.cos(input1.data, dtype=np.float64))

def cos(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.cos(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=cosGrad, prompt="Cosine")

def cosGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(-np.sin(input1.data, dtype=np.float64)) if not localDerivative is None else Tensor(-np.sin(input1.data, dtype=np.float64))

def tan(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.tan(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=tanGrad, prompt="Tangent")

def tanGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor((1 / np.cos(input1.data, dtype=np.float64)) ** 2) if not localDerivative is None else Tensor((1 / np.cos(input1.data, dtype=np.float64)) ** 2)

def csc(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(1 / np.sin(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=cscGrad, prompt="Cosecant")

def cscGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(-(1 / np.sin(input1.data, dtype=np.float64)) * (1 / np.tan(input1.data, dtype=np.float64))) if not localDerivative is None else Tensor(-(1 / np.sin(input1.data, dtype=np.float64)) * (1 / np.tan(input1.data, dtype=np.float64)))

def sec(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(1 / np.cos(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=secGrad, prompt="Secant")

def secGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor((1 / np.cos(input1.data, dtype=np.float64)) * (np.tan(input1.data, dtype=np.float64))) if not localDerivative is None else Tensor((1 / np.cos(input1.data, dtype=np.float64)) * (np.tan(input1.data, dtype=np.float64)))

def cot(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(1 / np.tan(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=cotGrad, prompt="Cotangent")

def cotGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(-((1 / np.sin(input1.data, dtype=np.float64)) ** 2)) if not localDerivative is None else Tensor(-((1 / np.sin(input1.data, dtype=np.float64)) ** 2))

def sum(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.sum(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=sumGrad, prompt="Summation")

def sumGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(np.ones_like(input1.data, dtype=np.float64)) if not localDerivative is None else Tensor(np.ones_like(input1.data, dtype=np.float64))

def abs(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.abs(x.data, dtype=np.float64))
    return func1(input1=input1, operation=operation, gradient=absGrad, prompt="Absolute Value")

def absGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(np.sign(input1.data, dtype=np.float64)) if not localDerivative is None else Tensor(np.sign(input1.data, dtype=np.float64))

def Sigmoid(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(1 / (1 + np.exp(-x)))
    return func1(input1=input1, operation=operation, gradient=SigmoidGrad, prompt="Sigmoid")

def SigmoidGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    sigmoid = Sigmoid(input1)
    return localDerivative * Tensor(sigmoid * (1 - sigmoid)) if not localDerivative is None else Tensor(sigmoid * (1 - sigmoid))

def Tanh(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.tanh(x))
    return func1(input1=input1, operation=operation, gradient=TanhGrad, prompt="Tanh")

def TanhGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(1 - (Tanh(input1) ** 2)) if not localDerivative is None else Tensor(1 - (Tanh(input1) ** 2))

def ReLU(input1: Any) -> Union[Node, Tensor]:
    operation = lambda x: Tensor(np.where(x.data < 0, 0, x.data))
    return func1(input1=input1, operation=operation, gradient=ReLUGrad, prompt="Rectified Linear Unit")

def ReLUGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    return localDerivative * Tensor(np.where(input1.data < 0, 0, 1)) if not localDerivative is None else Tensor(np.where(input1.data < 0, 0, 1))

def Softmax(input1: Any) -> Union[Node, Tensor]:
    def operation(x):
        exp = np.exp(x)
        return Tensor((1 / np.sum(exp)) * exp)
    return func1(input1=input1, operation=operation, gradient=SoftmaxGrad, prompt="Softmax")

def SoftmaxGrad(input1: Tensor, position: int, localDerivative: Tensor = None):
    softmax = Softmax(input1)
    return localDerivative * Tensor(softmax * (1 - softmax)) if not localDerivative is None else Tensor(softmax * (1 - softmax))


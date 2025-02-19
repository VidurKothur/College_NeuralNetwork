"""
This module contains the Node class, which is a wrapper for an element of the computational graph, and each
Node object contains important variables for automatic differentiation, including the identity (index within
the registry), the inputs to the node, the indices of every Node that can be traced back to this one somewhere
along the graph, the actual value that the Node carries, the respective operation and gradient function that
is called upon during backpropagation, and the debug parameter.

This module redirects operations to the functions module, just like the Tensor class, so that the appropriate
computational graph logic can be handled.

"""
class Node:
    def __init__(self, graph, identity, value, paths, inputs, gradFunction, debug, optimizer = None):
        self.value = value
        self.identity = identity
        self.paths = paths
        self.inputs = inputs
        self.gradFunction = gradFunction
        self.graph = graph
        self.debug = debug
        self.optimizer = optimizer

    def sendToGraph(self):
        self.graph.registry.append(self)

    def __repr__(self):
        return f"<Node {self.identity}: Value = {self.value}, Paths = {self.paths}, Inputs = {self.inputs} Func = {self.gradFunction.__name__ if not self.gradFunction is None else "None"}>"

    def __add__(self, value):
        import functions
        return functions.add(self, value)
    
    def __radd__(self, value):
        import functions
        return functions.add(value, self)
    
    def __sub__(self, value):
        import functions
        return functions.sub(self, value)
    
    def __rsub__(self, value):
        import functions
        return functions.sub(value, self)
    
    def __mul__(self, value):
        import functions
        return functions.mul(self, value)
    
    def __rmul__(self, value):
        import functions
        return functions.mul(value, self)
    
    def __pow__(self, value):
        import functions
        return functions.pow(self, value)
    
    def __rpow__(self, value):
        import functions
        return functions.pow(value, self)
    
    def __matmul__(self, value):
        import functions
        return functions.mmul(self, value)
    
    def __rmatmul__(self, value):
        import functions
        return functions.mmul(value, self)
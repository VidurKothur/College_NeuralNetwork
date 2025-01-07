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
        #Remove method when done debugging
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
    
    def __matmul__(self, value):
        import functions
        return functions.mmul(self, value)
    
    def __rmatmul__(self, value):
        import functions
        return functions.mmul(value, self)
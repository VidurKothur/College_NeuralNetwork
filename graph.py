"""
This function recursively computes the gradients of a final output node in a computational graph
with respect to specified input nodes, whose indices are given in the "targetNodeIndices" parameter.

The function is designed to compute multiple gradients at once, in an approach inspired by dynamic
programming, so that the entire function doesn't need to be run once for each gradient. This is made
possible since the program loops through every gradient that needs to be calculated, adds the appropriate
local derivative to the "localDerivatives" list, then uses this list along with results from recursively
calling a gradient on all the necessary root nodes, to return a final answer.

"""
def gradient(targetNodeIndices, computationGraph, localGradient = None, deg = 0):
    gradients = [0 for _ in range(len(targetNodeIndices))]
    localGradientChains = [[] for _ in range(len(computationGraph[-1].inputs))]
    localDerivatives = [None for _ in range(len(computationGraph[-1].inputs))]
    for idx in range(len(targetNodeIndices)):
        targetIndex = targetNodeIndices[idx]
        if targetIndex in computationGraph[-1].paths:
            if len(computationGraph) - 1 == targetIndex:
                gradients[idx] = 1 if localGradient is None else localGradient
            else:
                for inputIdx in range(len(computationGraph[-1].inputs)):
                    inputNodeIndex = computationGraph[-1].inputs[inputIdx]
                    if targetIndex in computationGraph[inputNodeIndex].paths:
                        localGradientChains[inputIdx].append(idx)
                        if localDerivatives[inputIdx] is None:
                            localDerivatives[inputIdx] = computationGraph[-1].gradFunction(*tuple(computationGraph[inp].value for inp in computationGraph[-1].inputs), inputIdx, localGradient if not localGradient is None else None)
    """
    If the computation graph's output node has no inputs, the recursion terminates.
    The "localGradientChains" list holds partial derivatives and the target indices dependent on each input path.
    """
    for inputIdx in range(len(localGradientChains)):
        localDerivative = localDerivatives[inputIdx]
        dependentTargetIndices = localGradientChains[inputIdx]
        if localDerivative is None:
            continue
        newTargetIndices = [targetNodeIndices[i] for i in dependentTargetIndices]
        partialGradients = gradient(newTargetIndices, computationGraph[:computationGraph[-1].inputs[inputIdx] + 1], localDerivative, deg + 1)
        for idx, targetIdx in enumerate(dependentTargetIndices):
            gradients[targetIdx] += partialGradients[idx]
    return gradients

"""
The Graph class holds a node list that represents the computational graph, self.registry. The self.gradients
variable holds a list of indices within the registry that contains learnable parameters, the self.built variable
determines whether a Graph object is holding a computational graph or not, and the self.counts variable contains a 
list of indices and keys for debugging purposes. The self.result variable stores the index within the registry 
that corresponds to the output of this specific function.

The runGradient method will run the gradient function with respect to the specific variables in the self.gradients
variable and run them through the optimizer to update their values (and print the appropriate value if necessary).

The runFunction method will return the value of the Node on the computational graph that corresponds to the output 
of the function, and the destroyFunction method will clear the self.registry variable and reset all other variables.

"""
class Graph:
    def __init__(self):
        self.registry = []
        self.gradients = []
        self.built = False
        self.counts = None
        self.result = None

    def __repr__(self):
        final = "----------------\nGraph:\n"
        for node in self.registry:
            final += f"{node.__repr__()}\n"
        final += "----------------"
        return final

    def runGradient(self, options):
        grads = gradient(self.gradients, self.registry[:self.result + 1])
        for b in range(len(self.counts)):
            debug = False
            for ind in self.counts[b][1]:
                if not self.registry[ind].debug is None:
                    if not debug:
                        print(f"{self.counts[b][0]}:\n----------------------------------------")
                        print("Parameters:\n--------------------")
                        debug = True
                    print(f"{self.registry[ind].debug}: {grads[self.gradients.index(ind)]}")
            if debug:
                print("--------------------")
                print("----------------------------------------\n")
                    
        for a in range(len(self.gradients)):
            self.registry[a].value = self.registry[a].optimizer(self.registry[a].value, grads[a])
    
    def runFunction(self):
        return self.registry[self.result].value

    def destroyFunction(self):
        self.registry = []
        self.gradients = []
        self.built = False
        self.result = None
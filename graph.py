import functions
import data

def gradient(gradients, registry, options, deg):
    final = [0 for _ in range(len(gradients))]
    bs = [[] for _ in range(len(registry[-1].inputs))]
    for b in range(len(gradients)):
        if gradients[b] in registry[-1].paths:
            if len(registry) - 1 == gradients[b]:
                final[b] = 1
            else:
                for c in range(len(registry[-1].inputs)):
                    if gradients[b] in registry[registry[-1].inputs[c]].paths:
                        chain = registry[-1].gradFunction(*tuple(registry[inp].value for inp in registry[-1].inputs), c, options)
                        bs[c].append((b, chain))
    for d in range(len(bs)):
        if not bs[d]:
            continue
        new_grads = [item[0] for item in bs[d]]
        grads = gradient(new_grads, registry[:registry[-1].inputs[d] + 1], options, deg + 1)
        for f, (index, chain) in enumerate(bs[d]):
            final[index] += chain * grads[f]
    return final

class Graph:
    def __init__(self):
        self.registry = []
        self.gradients = []
        self.built = False
        self.counts = None

    def __repr__(self):
        #Remove method when done debugging
        final = "----------------\nGraph:\n"
        for node in self.registry:
            final += f"{node.__repr__()}\n"
        final += "----------------"
        return final

    def runGradient(self, options):
        grads = gradient(self.gradients, self.registry, options, 1)
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
    
    def destroyFunction(self):
        self.registry = []
        self.gradients = []
        self.built = False
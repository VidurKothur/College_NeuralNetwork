import numpy as np
from typing import *
import copy

"""
This is a custom data type that reroutes operations to the functions module, where it can be used in conjuction with 
the Node data type.

The purpose of this class is to wrap the numpy ndarray class, allowing numpy arrays to interact with Node objects while
avoiding a central problem: the np.ndarray class already implements methods like __add__, __sub__, etc., which I found to
change the data stored within that array in some cases. When I tried to, for example, add an np.ndarray and Node object,
even though the Node class has __radd__ implemented, the computation defaults to numpy's __add__ method, changing the data
in my np.ndarray. By computing with Tensors and Nodes, both Tensor + Node and Node + Tensor ultimately defers the computation
to the functions module, where computational graph logic is handled.

"""

class Tensor:
    def __init__(self, iterable: Any) -> None:
        try:
            if isinstance(iterable, Tensor):
                self.data = copy.deepcopy(iterable.data)
            else:
                self.data = np.array(iterable, dtype=np.float64)
        except:
            raise TypeError(f"Error: The input 'iterable' of value {iterable} of type {type(iterable)} could not be resolved to a mathematical object")

    def __repr__(self) -> str:
        return f"\n{self.data.__str__()}\n"
    
    def __str__(self) -> str:
        return f"\n{self.data.__str__()}\n"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self) -> iter:
        if self.data.ndim > 0:
            self.iteration = 0
            return self
        else:
            raise Exception("Error: Iteration is not supported on a zero-dimensional object")

    def __next__(self):
        if self.iteration < len(self.data):
            self.iteration += 1
            return Tensor(self.data[self.iteration - 1])
        else:
            raise StopIteration
    
    def __getattr__(self, name: str):
        toRet = self.data.__getattribute__(name)
        if isinstance(toRet, np.ndarray):
            return Tensor(toRet)
        else:
            return toRet
        
    def __getitem__(self, key: Any):
        toRet = self.data[key]
        if isinstance(toRet, np.ndarray):
            return Tensor(toRet)
        else:
            return toRet
            
    def __neg__(self):
        return Tensor(-self.data)
    
    def __pos__(self):
        return Tensor(self.data)
    
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
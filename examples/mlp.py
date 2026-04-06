"""Example: Multi-layer perceptrons built with mytorch."""

import numpy as np
from mytorch.nn import Linear, ReLU


class MLP0:
    """Single linear layer (2 -> 3) with ReLU."""

    def __init__(self):
        self.layers = [Linear(2, 3), ReLU()]

    def forward(self, A):
        Z = self.layers[0].forward(A)
        return self.layers[1].forward(Z)

    def backward(self, dLdA):
        dLdZ = self.layers[1].backward(dLdA)
        return self.layers[0].backward(dLdZ)


class MLP4:
    """Four hidden layers: 2 -> 4 -> 8 -> 8 -> 4 -> 2, all ReLU."""

    def __init__(self):
        self.layers = [
            Linear(2, 4), ReLU(),
            Linear(4, 8), ReLU(),
            Linear(8, 8), ReLU(),
            Linear(8, 4), ReLU(),
            Linear(4, 2), ReLU(),
        ]

    def forward(self, A):
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dLdA):
        for layer in reversed(self.layers):
            dLdA = layer.backward(dLdA)
        return dLdA

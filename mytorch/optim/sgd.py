import numpy as np


class SGD:
    """Stochastic Gradient Descent with optional momentum.

    Updates every layer in model.layers that has weights (.W, .b, .dLdW, .dLdb);
    parameter-free layers such as activations are skipped.
    """

    def __init__(self, model, lr=0.1, momentum=0):
        self.l = [layer for layer in model.layers if hasattr(layer, "W")]
        self.L = len(self.l)
        self.lr = lr
        self.mu = momentum

        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        for i in range(self.L):
            if self.mu == 0:
                self.l[i].W -= self.lr * self.l[i].dLdW
                self.l[i].b -= self.lr * self.l[i].dLdb
            else:
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb
                self.l[i].W -= self.lr * self.v_W[i]
                self.l[i].b -= self.lr * self.v_b[i]

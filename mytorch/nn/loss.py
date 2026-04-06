import numpy as np


class MSELoss:
    """Mean Squared Error loss."""

    def forward(self, A, Y):
        """
        :param A: predictions (N, C)
        :param Y: targets (N, C)
        :return: scalar MSE loss
        """
        self.A = A
        self.Y = Y
        self.N, self.C = A.shape
        return np.sum((A - Y) ** 2) / (self.N * self.C)

    def backward(self):
        return 2.0 * (self.A - self.Y) / (self.N * self.C)


class CrossEntropyLoss:
    """Cross-entropy loss with built-in softmax (like PyTorch's CrossEntropyLoss)."""

    def forward(self, A, Y):
        """
        :param A: raw logits (N, C)
        :param Y: one-hot targets (N, C)
        :return: scalar cross-entropy loss
        """
        self.A = A
        self.Y = Y
        self.N, self.C = A.shape

        Z_shift = A - np.max(A, axis=1, keepdims=True)
        expZ = np.exp(Z_shift)
        self.softmax = expZ / np.sum(expZ, axis=1, keepdims=True)

        eps = 1e-12
        crossentropy = -np.sum(self.Y * np.log(self.softmax + eps), axis=1)
        return np.sum(crossentropy) / self.N

    def backward(self):
        return (self.softmax - self.Y) / self.N

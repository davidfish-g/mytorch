import numpy as np


class Linear:
    """Fully-connected layer supporting arbitrary batch dimensions.

    Forward:  Z = A @ W^T + b
    Backward: computes dL/dA, dL/dW, dL/db
    """

    def __init__(self, in_features, out_features):
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)

    def init_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: (*, in_features)
        :return:  (*, out_features)
        """
        self.A = A
        self.input_shape = A.shape

        A_flat = A.reshape(-1, self.W.shape[1])
        Z_flat = A_flat @ self.W.T + self.b

        output_shape = self.input_shape[:-1] + (self.W.shape[0],)
        return Z_flat.reshape(output_shape)

    def backward(self, dLdZ):
        """
        :param dLdZ: (*, out_features)
        :return:     (*, in_features)
        """
        dLdZ_flat = dLdZ.reshape(-1, self.W.shape[0])
        A_flat = self.A.reshape(-1, self.W.shape[1])

        self.dLdW = dLdZ_flat.T @ A_flat
        self.dLdb = np.sum(dLdZ_flat, axis=0)
        self.dLdA = (dLdZ_flat @ self.W).reshape(self.input_shape)

        return self.dLdA

    def __call__(self, A):
        return self.forward(A)

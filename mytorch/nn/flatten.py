import numpy as np


class Flatten:
    def forward(self, A):
        """
        :param A: (batch_size, *spatial_dims)
        :return:  (batch_size, flattened_dim)
        """
        self._shape = A.shape
        return A.reshape(A.shape[0], -1)

    def backward(self, dLdZ):
        return dLdZ.reshape(self._shape)

import numpy as np


class BatchNorm1d:
    """Batch normalization over a mini-batch of 2-D inputs (N, C).

    Tracks running mean / variance for inference via exponential moving average.
    """

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        :param Z: (N, num_features)
        :param eval: if True, use running stats instead of batch stats
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = np.mean(Z, axis=0, keepdims=True)
        self.V = np.mean((Z - self.M) ** 2, axis=0, keepdims=True)

        if not eval:
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
            self.running_M = self.alpha * self.running_M + (1.0 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1.0 - self.alpha) * self.V
        else:
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)

        self.BZ = self.BW * self.NZ + self.Bb
        return self.BZ

    def backward(self, dLdBZ):
        """
        :param dLdBZ: (N, num_features)
        :return: dLdZ (N, num_features)
        """
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW
        inv_std = 1.0 / np.sqrt(self.V + self.eps)

        dLdV = np.sum(dLdNZ * (self.Z - self.M) * (-0.5) * (inv_std ** 3), axis=0, keepdims=True)

        sum_diff = np.sum(self.Z - self.M, axis=0, keepdims=True)
        dNZdM = -inv_std + (self.Z - self.M) * (inv_std ** 3) * (1.0 / self.N) * sum_diff
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True)

        dLdZ = dLdNZ * inv_std + dLdV * (2.0 / self.N) * (self.Z - self.M) + (1.0 / self.N) * dLdM
        return dLdZ

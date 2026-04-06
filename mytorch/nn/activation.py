import numpy as np
import scipy.special


class Identity:
    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self, dLdA):
        return dLdA * np.ones(self.A.shape, dtype="f")


class Sigmoid:
    def forward(self, Z):
        self.A = 1.0 / (1.0 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        return dLdA * self.A * (1.0 - self.A)


class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA, state=None):
        """Supports optional external state for BPTT in RNNs."""
        a = state if state is not None else self.A
        return dLdA * (1.0 - a * a)


class ReLU:
    def forward(self, Z):
        self.A = np.maximum(0.0, Z)
        return self.A

    def backward(self, dLdA):
        return dLdA * np.where(self.A > 0.0, 1.0, 0.0)


class GELU:
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1.0 + scipy.special.erf(Z / np.sqrt(2.0)))
        return self.A

    def backward(self, dLdA):
        Z = self.Z
        dAdZ = (
            0.5 * (1.0 + scipy.special.erf(Z / np.sqrt(2.0)))
            + (Z / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * Z * Z)
        )
        return dLdA * dAdZ


class Swish:
    """SiLU / Swish with learnable beta parameter."""

    def __init__(self, beta=1.0):
        self.beta = float(beta)

    def forward(self, Z):
        self.Z = Z
        self.sigma = 1.0 / (1.0 + np.exp(-self.beta * Z))
        self.A = Z * self.sigma
        return self.A

    def backward(self, dLdA):
        sigma = self.sigma
        dAdZ = sigma + self.beta * self.Z * sigma * (1.0 - sigma)
        self.dLdbeta = np.sum(dLdA * (self.Z * self.Z * sigma * (1.0 - sigma)))
        return dLdA * dAdZ


class Softmax:
    """Generic softmax supporting arbitrary dimensions.

    The simple 2-D (N, C) case uses the Jacobian formulation.
    Higher-dimensional inputs are reshaped and handled efficiently.
    """

    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, Z):
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("dim out of range for input shape")

        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        Z_exp = np.exp(Z_shifted)
        self.A = Z_exp / np.sum(Z_exp, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        shape = self.A.shape
        C = shape[self.dim]

        if len(shape) > 2:
            self.A = np.moveaxis(self.A, self.dim, -1).reshape(-1, C)
            dLdA = np.moveaxis(dLdA, self.dim, -1).reshape(-1, C)

        calc_axis = 1 if len(shape) > 2 else self.dim
        grad_dot = np.sum(dLdA * self.A, axis=calc_axis, keepdims=True)
        dLdZ = self.A * (dLdA - grad_dot)

        if len(shape) > 2:
            axis = self.dim if self.dim >= 0 else len(shape) + self.dim
            permuted_shape = list(shape)
            permuted_shape.pop(axis)
            permuted_shape.append(C)
            self.A = np.moveaxis(self.A.reshape(permuted_shape), -1, self.dim)
            dLdZ = np.moveaxis(dLdZ.reshape(permuted_shape), -1, self.dim)

        return dLdZ

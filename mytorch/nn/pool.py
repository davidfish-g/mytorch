import numpy as np

from .resampling import Downsample2d


class MaxPool2d_stride1:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        :param A: (batch_size, channels, height, width)
        :return:  (batch_size, channels, out_height, out_width)
        """
        self.A = A
        N, C, H, W = A.shape
        K = self.kernel
        out_h, out_w = H - K + 1, W - K + 1
        Z = np.empty((N, C, out_h, out_w))

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        Z[n, c, i, j] = np.max(A[n, c, i:i+K, j:j+K])
        return Z

    def backward(self, dLdZ):
        N, C, H, W = self.A.shape
        K = self.kernel
        out_h, out_w = H - K + 1, W - K + 1
        dLdA = np.zeros_like(self.A)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = self.A[n, c, i:i+K, j:j+K]
                        mask = (patch == patch.max())
                        dLdA[n, c, i:i+K, j:j+K] += mask * (dLdZ[n, c, i, j] / mask.sum())
        return dLdA


class MeanPool2d_stride1:
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        :param A: (batch_size, channels, height, width)
        :return:  (batch_size, channels, out_height, out_width)
        """
        self.A = A
        N, C, H, W = A.shape
        K = self.kernel
        out_h, out_w = H - K + 1, W - K + 1
        area = K * K
        Z = np.empty((N, C, out_h, out_w))

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        Z[n, c, i, j] = np.sum(A[n, c, i:i+K, j:j+K]) / area
        return Z

    def backward(self, dLdZ):
        N, C, H, W = self.A.shape
        K = self.kernel
        out_h, out_w = H - K + 1, W - K + 1
        scale = 1.0 / (K * K)
        dLdA = np.zeros_like(self.A)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        dLdA[n, c, i:i+K, j:j+K] += scale * dLdZ[n, c, i, j]
        return dLdA


class MaxPool2d:
    def __init__(self, kernel, stride):
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        return self.downsample2d.forward(self.maxpool2d_stride1.forward(A))

    def backward(self, dLdZ):
        return self.maxpool2d_stride1.backward(self.downsample2d.backward(dLdZ))


class MeanPool2d:
    def __init__(self, kernel, stride):
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        return self.downsample2d.forward(self.meanpool2d_stride1.forward(A))

    def backward(self, dLdZ):
        return self.meanpool2d_stride1.backward(self.downsample2d.backward(dLdZ))

import numpy as np

from .resampling import Downsample1d, Downsample2d, Upsample1d, Upsample2d


# ---------------------------------------------------------------------------
# 1-D Convolution
# ---------------------------------------------------------------------------

class Conv1d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        self.b = np.zeros(out_channels) if bias_init_fn is None else bias_init_fn(out_channels)
        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        :param A: (batch_size, in_channels, input_size)
        :return:  (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_size))

        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(output_size):
                    Z[i, j, k] = np.sum(self.W[j] * A[i, :, k:k + self.kernel_size]) + self.b[j]
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: (batch_size, out_channels, output_size)
        :return:     (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        dLdA = np.zeros_like(self.A)
        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

        for i in range(batch_size):
            for j in range(out_channels):
                self.dLdb[j] += np.sum(dLdZ[i, j, :])
                for k in range(output_size):
                    self.dLdW[j] += dLdZ[i, j, k] * self.A[i, :, k:k + self.kernel_size]
                    for c in range(self.in_channels):
                        dLdA[i, c, k:k + self.kernel_size] += self.W[j, c, :] * dLdZ[i, j, k]
        return dLdA


class Conv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                              weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        :param A: (batch_size, in_channels, input_size)
        :return:  (batch_size, out_channels, output_size)
        """
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode='constant')
        Z = self.conv1d_stride1.forward(A_padded)
        return self.downsample1d.forward(Z)

    def backward(self, dLdZ):
        dLdZ_up = self.downsample1d.backward(dLdZ)
        dLdA_padded = self.conv1d_stride1.backward(dLdZ_up)

        if self.pad > 0:
            return dLdA_padded[:, :, self.pad:-self.pad]
        return dLdA_padded


# ---------------------------------------------------------------------------
# 2-D Convolution
# ---------------------------------------------------------------------------

class Conv2d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        self.b = np.zeros(out_channels) if bias_init_fn is None else bias_init_fn(out_channels)
        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        :param A: (batch_size, in_channels, input_height, input_width)
        :return:  (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        N, C_in, H_in, W_in = A.shape
        K = self.kernel_size
        out_h = H_in - K + 1
        out_w = W_in - K + 1
        Z = np.empty((N, self.out_channels, out_h, out_w))

        for n in range(N):
            for f in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        Z[n, f, i, j] = np.sum(A[n, :, i:i+K, j:j+K] * self.W[f]) + self.b[f]
        return Z

    def backward(self, dLdZ):
        N, C_in, H_in, W_in = self.A.shape
        K = self.kernel_size
        out_h = H_in - K + 1
        out_w = W_in - K + 1

        self.dLdW = np.zeros_like(self.W)
        self.dLdb = dLdZ.sum(axis=(0, 2, 3))
        dLdA = np.zeros_like(self.A)

        for n in range(N):
            for f in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        grad = dLdZ[n, f, i, j]
                        self.dLdW[f] += self.A[n, :, i:i+K, j:j+K] * grad
                        dLdA[n, :, i:i+K, j:j+K] += self.W[f] * grad
        return dLdA


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.pad = padding
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                                              weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        :param A: (batch_size, in_channels, input_height, input_width)
        :return:  (batch_size, out_channels, output_height, output_width)
        """
        if self.pad > 0:
            A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        Z = self.conv2d_stride1.forward(A)
        return self.downsample2d.forward(Z)

    def backward(self, dLdZ):
        dLdZ_up = self.downsample2d.backward(dLdZ)
        dLdA_padded = self.conv2d_stride1.backward(dLdZ_up)
        self.dLdW = self.conv2d_stride1.dLdW
        self.dLdb = self.conv2d_stride1.dLdb

        if self.pad > 0:
            return dLdA_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return dLdA_padded


# ---------------------------------------------------------------------------
# Transposed Convolutions
# ---------------------------------------------------------------------------

class ConvTranspose1d:
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        self.upsampling_factor = upsampling_factor
        self.upsample1d = Upsample1d(upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                              weight_init_fn, bias_init_fn)

    def forward(self, A):
        return self.conv1d_stride1.forward(self.upsample1d.forward(A))

    def backward(self, dLdZ):
        delta = self.conv1d_stride1.backward(dLdZ)
        self.dLdW = self.conv1d_stride1.dLdW
        self.dLdb = self.conv1d_stride1.dLdb
        return self.upsample1d.backward(delta)


class ConvTranspose2d:
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        self.upsampling_factor = upsampling_factor
        self.upsample2d = Upsample2d(upsampling_factor)
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                                              weight_init_fn, bias_init_fn)

    def forward(self, A):
        return self.conv2d_stride1.forward(self.upsample2d.forward(A))

    def backward(self, dLdZ):
        delta = self.conv2d_stride1.backward(dLdZ)
        self.dLdW = self.conv2d_stride1.dLdW
        self.dLdb = self.conv2d_stride1.dLdb
        return self.upsample2d.backward(delta)

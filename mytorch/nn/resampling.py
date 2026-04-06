import numpy as np


class Upsample1d:
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        :param A: (batch_size, in_channels, input_width)
        :return:  (batch_size, in_channels, output_width)
        """
        k = self.upsampling_factor
        N, C, W_in = A.shape
        W_out = k * (W_in - 1) + 1
        Z = np.zeros((N, C, W_out), dtype=A.dtype)
        Z[:, :, ::k] = A
        return Z

    def backward(self, dLdZ):
        return dLdZ[:, :, ::self.upsampling_factor]


class Downsample1d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        :param A: (batch_size, in_channels, input_width)
        :return:  (batch_size, in_channels, output_width)
        """
        self._input_shape = A.shape
        return A[:, :, ::self.downsampling_factor]

    def backward(self, dLdZ):
        N, C, W_in = self._input_shape
        dLdA = np.zeros((N, C, W_in), dtype=dLdZ.dtype)
        dLdA[:, :, ::self.downsampling_factor] = dLdZ
        return dLdA


class Upsample2d:
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        :param A: (batch_size, in_channels, input_height, input_width)
        :return:  (batch_size, in_channels, output_height, output_width)
        """
        k = self.upsampling_factor
        N, C, H_in, W_in = A.shape
        H_out = k * (H_in - 1) + 1
        W_out = k * (W_in - 1) + 1
        Z = np.zeros((N, C, H_out, W_out), dtype=A.dtype)
        Z[:, :, ::k, ::k] = A
        return Z

    def backward(self, dLdZ):
        k = self.upsampling_factor
        return dLdZ[:, :, ::k, ::k]


class Downsample2d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        :param A: (batch_size, in_channels, input_height, input_width)
        :return:  (batch_size, in_channels, output_height, output_width)
        """
        self._input_shape = A.shape
        k = self.downsampling_factor
        return A[:, :, ::k, ::k]

    def backward(self, dLdZ):
        N, C, H_in, W_in = self._input_shape
        k = self.downsampling_factor
        dLdA = np.zeros((N, C, H_in, W_in), dtype=dLdZ.dtype)
        dLdA[:, :, ::k, ::k] = dLdZ
        return dLdA

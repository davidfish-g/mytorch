import numpy as np

from .activation import Tanh


class RNNCell:
    """Vanilla RNN cell with forward and backward (BPTT) support.

    h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = Tanh()

        h, d = hidden_size, input_size
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        h, d = self.hidden_size, self.input_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        :param x:        (batch_size, input_size)
        :param h_prev_t: (batch_size, hidden_size)
        :return h_t:     (batch_size, hidden_size)
        """
        a = x @ self.W_ih.T + self.b_ih + h_prev_t @ self.W_hh.T + self.b_hh
        h_t = self.activation.forward(a)
        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        :param delta:    (batch_size, hidden_size) gradient from above
        :param h_t:      hidden state at current time step
        :param h_prev_l: input from previous layer at current time step
        :param h_prev_t: hidden state at previous time step
        :return: (dx, dh_prev_t)
        """
        batch_size = delta.shape[0]
        dz = delta * (1 - h_t ** 2)

        self.dW_ih += (dz.T @ h_prev_l) / batch_size
        self.dW_hh += (dz.T @ h_prev_t) / batch_size
        self.db_ih += dz.mean(axis=0)
        self.db_hh += dz.mean(axis=0)

        dx = dz @ self.W_ih
        dh_prev_t = dz @ self.W_hh
        return dx, dh_prev_t

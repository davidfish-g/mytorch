import numpy as np

from .activation import Sigmoid, Tanh


class GRUCell:
    """Gated Recurrent Unit cell with forward and backward support.

    Gates:
        r_t = sigmoid(W_rx @ x_t + b_rx + W_rh @ h_{t-1} + b_rh)   (reset)
        z_t = sigmoid(W_zx @ x_t + b_zx + W_zh @ h_{t-1} + b_zh)   (update)
        n_t = tanh(W_nx @ x_t + b_nx + r_t * (W_nh @ h_{t-1} + b_nh))
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h, d = hidden_size, input_size

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d)); self.dWzx = np.zeros((h, d)); self.dWnx = np.zeros((h, d))
        self.dWrh = np.zeros((h, h)); self.dWzh = np.zeros((h, h)); self.dWnh = np.zeros((h, h))
        self.dbrx = np.zeros(h); self.dbzx = np.zeros(h); self.dbnx = np.zeros(h)
        self.dbrh = np.zeros(h); self.dbzh = np.zeros(h); self.dbnh = np.zeros(h)

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx; self.Wzx = Wzx; self.Wnx = Wnx
        self.Wrh = Wrh; self.Wzh = Wzh; self.Wnh = Wnh
        self.brx = brx; self.bzx = bzx; self.bnx = bnx
        self.brh = brh; self.bzh = bzh; self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        :param x:        (input_size,)
        :param h_prev_t: (hidden_size,)
        :return h_t:     (hidden_size,)
        """
        self.x = x
        self.hidden = h_prev_t

        self.r = self.r_act.forward(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act.forward(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        self.u = self.Wnh @ h_prev_t + self.bnh
        self.n = self.h_act.forward(self.Wnx @ x + self.bnx + self.r * self.u)
        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        return h_t

    def backward(self, delta):
        """
        :param delta: (hidden_size,) gradient from above
        :return: (dx, dh_prev_t)
        """
        h_prev = self.hidden
        r, z, n, u = self.r, self.z, self.n, self.u

        dnt = delta * (1 - z)
        dzt = delta * (h_prev - n)
        dh_prev_from_ht = delta * z

        da_n = dnt * (1 - n ** 2)
        self.dWnx += np.outer(da_n, self.x)
        self.dbnx += da_n
        dx_from_n = self.Wnx.T @ da_n

        du = da_n * r
        dr_from_n = da_n * u
        self.dWnh += np.outer(du, h_prev)
        self.dbnh += du
        dh_prev_from_n = self.Wnh.T @ du

        da_z = dzt * z * (1 - z)
        self.dWzx += np.outer(da_z, self.x)
        self.dbzx += da_z
        self.dWzh += np.outer(da_z, h_prev)
        self.dbzh += da_z
        dx_from_z = self.Wzx.T @ da_z
        dh_prev_from_z = self.Wzh.T @ da_z

        da_r = dr_from_n * r * (1 - r)
        self.dWrx += np.outer(da_r, self.x)
        self.dbrx += da_r
        self.dWrh += np.outer(da_r, h_prev)
        self.dbrh += da_r
        dx_from_r = self.Wrx.T @ da_r
        dh_prev_from_r = self.Wrh.T @ da_r

        dx = dx_from_n + dx_from_z + dx_from_r
        dh_prev_t = dh_prev_from_ht + dh_prev_from_n + dh_prev_from_z + dh_prev_from_r

        return dx, dh_prev_t

import numpy as np

from .activation import Softmax
from .linear import Linear


class ScaledDotProductAttention:
    """Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V"""

    def __init__(self):
        self.eps = 1e10
        self.softmax = Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """
        :param Q:    (N, ..., H, L, E)
        :param K:    (N, ..., H, S, E)
        :param V:    (N, ..., H, S, Ev)
        :param mask: boolean, True = ignore position
        :return:     (N, ..., H, L, Ev)
        """
        self.Q, self.K, self.V = Q, K, V
        d_k = Q.shape[-1]

        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, scores - self.eps, scores)

        self.attention_scores = self.softmax.forward(scores)
        return np.matmul(self.attention_scores, V)

    def backward(self, d_output):
        """
        :param d_output: (N, ..., H, L, Ev)
        :return: (dQ, dK, dV)
        """
        d_k = self.Q.shape[-1]

        d_V = np.matmul(np.swapaxes(self.attention_scores, -1, -2), d_output)
        d_attn = np.matmul(d_output, np.swapaxes(self.V, -1, -2))
        d_scores = self.softmax.backward(d_attn) / np.sqrt(d_k)

        d_Q = np.matmul(d_scores, self.K)
        d_K = np.matmul(np.swapaxes(d_scores, -1, -2), self.Q)
        return d_Q, d_K, d_V


class MultiHeadAttention:
    """Multi-head attention matching the PyTorch nn.MultiheadAttention interface."""

    def __init__(self, embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention()

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key:   (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S), True = ignore
        :param attn_mask:        (L, S), True = ignore
        :return: (N, L, E)
        """
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        q = self._split_heads(self.q_proj.forward(query))
        k = self._split_heads(self.k_proj.forward(key))
        v = self._split_heads(self.v_proj.forward(value))

        mask = self._merge_masks(key_padding_mask, attn_mask)
        attn_out = self.attention.forward(q, k, v, mask=mask)

        return self.out_proj.forward(self._concat_heads(attn_out))

    def backward(self, d_output):
        """
        :param d_output: (N, L, E)
        :return: (d_query, d_key, d_value) each (N, *, E)
        """
        d_concat = self.out_proj.backward(d_output)
        d_q, d_k, d_v = self.attention.backward(self._split_heads(d_concat))

        d_q = self.q_proj.backward(self._concat_heads(d_q))
        d_k = self.k_proj.backward(self._concat_heads(d_k))
        d_v = self.v_proj.backward(self._concat_heads(d_v))
        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        if key_padding_mask is None and attn_mask is None:
            return None
        key_mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else False
        att_mask = attn_mask[None, None, :, :] if attn_mask is not None else False
        combined = np.logical_or(key_mask, att_mask)
        return combined if isinstance(combined, np.ndarray) else None

    def _split_heads(self, x):
        """(N, seq, embed_dim) -> (N, num_heads, seq, head_dim)"""
        head_dim = self.embed_dim // self.num_heads
        return np.transpose(x.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim), (0, 2, 1, 3))

    def _concat_heads(self, x):
        """(N, num_heads, seq, head_dim) -> (N, seq, embed_dim)"""
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(x.shape[0], x.shape[1], self.embed_dim)

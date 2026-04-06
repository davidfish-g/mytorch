"""Example: RNN phoneme classifier and GRU character predictor built with mytorch."""

import numpy as np
from mytorch.nn import RNNCell, GRUCell, Linear


class RNNPhonemeClassifier:
    """Multi-layer RNN for phoneme classification.

    Stacks multiple RNN cells, runs over a sequence, then projects the final
    hidden state to class logits.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = [
            RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)
        self.hiddens = []

    def forward(self, x, h_0=None):
        """
        :param x:   (batch_size, seq_len, input_size)
        :param h_0: (num_layers, batch_size, hidden_size) or None
        :return: logits (batch_size, output_size)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        hidden = np.zeros((self.num_layers, batch_size, self.hidden_size)) if h_0 is None else h_0.copy()

        self.x = x
        self.hiddens = [hidden.copy()]

        for t in range(seq_len):
            for l in range(self.num_layers):
                h_prev_l = x[:, t, :] if l == 0 else hidden[l - 1]
                hidden[l] = self.rnn[l](h_prev_l, hidden[l])
            self.hiddens.append(hidden.copy())

        return self.output_layer.forward(hidden[-1])

    def backward(self, delta):
        """
        :param delta: (batch_size, output_size) gradient from loss
        :return: dh_0 (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size))
        dh[-1] = self.output_layer.backward(delta)

        for t in range(seq_len - 1, -1, -1):
            for l in range(self.num_layers - 1, -1, -1):
                h_t = self.hiddens[t + 1][l]
                h_prev_t = self.hiddens[t][l]
                h_prev_l = self.x[:, t, :] if l == 0 else self.hiddens[t + 1][l - 1]

                dx, dh_prev_t = self.rnn[l].backward(dh[l], h_t, h_prev_l, h_prev_t)
                dh[l] = dh_prev_t
                if l != 0:
                    dh[l - 1] += dx

        return dh / batch_size


class CharacterPredictor:
    """Single GRU cell + linear projection for character-level prediction."""

    def __init__(self, input_dim, hidden_dim, num_classes):
        self.gru = GRUCell(input_dim, hidden_dim)
        self.projection = Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x, h):
        """
        :param x: (input_dim,) single time step
        :param h: (hidden_dim,) previous hidden state
        :return: (logits, h_next)
        """
        h_next = self.gru(x, h)
        logits = self.projection(h_next.reshape(1, -1)).reshape(-1)
        return logits, h_next

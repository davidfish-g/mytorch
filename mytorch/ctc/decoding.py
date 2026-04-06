import numpy as np


class GreedySearchDecoder:
    """Greedy (best-path) CTC decoder."""

    def __init__(self, symbol_set):
        """
        :param symbol_set: list of symbols (vocabulary without blank)
        """
        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """
        :param y_probs: (num_symbols + 1, seq_length) or (num_symbols + 1, seq_length, batch_size)
        :return: (decoded_path, path_prob)
        """
        blank = 0
        path_prob = 1

        if y_probs.ndim == 3:
            probs = y_probs[:, :, 0]
        else:
            probs = y_probs

        T = probs.shape[1]
        indices = []
        for t in range(T):
            col = probs[:, t]
            idx = int(np.argmax(col))
            path_prob *= float(col[idx])
            indices.append(idx)

        compressed = []
        prev = None
        for idx in indices:
            if idx != blank and idx != prev:
                compressed.append(idx)
            prev = idx

        decoded_path = "".join(self.symbol_set[i - 1] for i in compressed)
        return decoded_path, path_prob

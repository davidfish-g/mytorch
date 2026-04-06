import numpy as np


class CTC:
    """Core CTC forward-backward algorithm."""

    def __init__(self, BLANK=0):
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Insert blanks between and around target symbols.

        :param target: (target_len,) e.g. [B, IY, IY, F]
        :return: (extended_symbols, skip_connect)
                 e.g. [-, B, -, IY, -, IY, -, F, -]
        """
        extended = [self.BLANK]
        for symbol in target:
            extended.append(symbol)
            extended.append(self.BLANK)

        N = len(extended)
        skip_connect = np.zeros(N, dtype=int)
        for i in range(2, N):
            if extended[i] != self.BLANK and extended[i] != extended[i - 2]:
                skip_connect[i] = 1

        return extended, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """
        :param logits: (input_len, num_symbols) emission probabilities
        :return: alpha (input_len, 2 * target_len + 1)
        """
        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros((T, S))

        alpha[0, 0] = logits[0, extended_symbols[0]]
        if S > 1:
            alpha[0, 1] = logits[0, extended_symbols[1]]

        for t in range(1, T):
            for s in range(S):
                total = alpha[t - 1, s]
                if s - 1 >= 0:
                    total += alpha[t - 1, s - 1]
                if s - 2 >= 0 and skip_connect[s]:
                    total += alpha[t - 1, s - 2]
                alpha[t, s] = total * logits[t, extended_symbols[s]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """
        :return: beta (input_len, 2 * target_len + 1)
        """
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros((T, S))

        beta[T - 1, S - 1] = 1.0
        if S > 1:
            beta[T - 1, S - 2] = 1.0

        for t in range(T - 2, -1, -1):
            for s in range(S):
                total = beta[t + 1, s] * logits[t + 1, extended_symbols[s]]
                if s + 1 < S:
                    total += beta[t + 1, s + 1] * logits[t + 1, extended_symbols[s + 1]]
                if s + 2 < S and skip_connect[s + 2]:
                    total += beta[t + 1, s + 2] * logits[t + 1, extended_symbols[s + 2]]
                beta[t, s] = total

        return beta

    def get_posterior_probs(self, alpha, beta):
        """
        :return: gamma (input_len, 2 * target_len + 1)
        """
        gamma = alpha * beta
        eps = 1e-8
        for t in range(gamma.shape[0]):
            gamma[t, :] /= (np.sum(gamma[t, :]) + eps)
        return gamma


class CTCLoss:
    """CTC loss function with forward and backward pass."""

    def __init__(self, BLANK=0):
        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC(BLANK)

    def __call__(self, logits, target, input_lengths, target_lengths):
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """
        :param logits:         (seq_length, batch_size, num_symbols)
        :param target:         (batch_size, padded_target_len)
        :param input_lengths:  (batch_size,)
        :param target_lengths: (batch_size,)
        :return: scalar loss (mean over batch)
        """
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        B = target.shape[0]
        total_loss = np.zeros(B)
        self.extended_symbols = []
        eps = 1e-8

        for b in range(B):
            tgt = target[b][:target_lengths[b]]
            lgt = logits[:input_lengths[b], b]

            extended, skip = self.ctc.extend_target_with_blank(tgt)
            alpha = self.ctc.get_forward_probs(lgt, extended, skip)
            beta = self.ctc.get_backward_probs(lgt, extended, skip)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            self.gammas.append(gamma)
            self.extended_symbols.append(extended)
            total_loss[b] = -np.sum(gamma * np.log(lgt[:, extended] + eps))

        return np.sum(total_loss) / B

    def backward(self):
        """
        :return: dY (seq_length, batch_size, num_symbols)
        """
        T, B, C = self.logits.shape
        dY = np.zeros_like(self.logits)
        eps = 1e-8

        for b in range(B):
            extended, _ = self.ctc.extend_target_with_blank(
                self.target[b][:self.target_lengths[b]]
            )
            gamma = self.gammas[b]
            logit = self.logits[:self.input_lengths[b], b]
            inp_len = self.input_lengths[b]

            for t in range(inp_len):
                for r in range(len(extended)):
                    c = extended[r]
                    dY[t, b, c] -= gamma[t, r] / (logit[t, c] + eps)

        return dY

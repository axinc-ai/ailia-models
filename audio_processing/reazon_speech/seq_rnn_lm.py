import numpy as np
from scipy.special import log_softmax

onnx = False


class SequentialRNNLM(object):
    def __init__(
            self, net):
        self.net = net
        self.nhid = 2024

    def select_state(self, state, i: int, new_id: int = None):
        """Select state with relative ids in the main beam search.
        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label index to select a state if necessary
        Returns:
            state: pruned state
        """
        return None if state is None else state[i]

    def batch_score(
            self, ys, states, xs):
        if states[0] is None:
            h = np.zeros((2, ys.shape[0], self.nhid), dtype=np.float32)
            c = np.zeros((2, ys.shape[0], self.nhid), dtype=np.float32)
        else:
            h = np.stack([h for h, c in states], axis=1)
            c = np.stack([c for h, c in states], axis=1)

        # feedforward
        ys = ys[:, -1:].astype(int)
        if not onnx:
            output = self.net.predict([ys, h, c])
        else:
            output = self.net.run(None, {'input': ys, 'h_0': h, 'c_0': c})
        ys, h, c = output

        ys = ys[:, 0, :]
        logp = log_softmax(ys, axis=-1)

        states = [(h[:, i], c[:, i]) for i in range(h.shape[1])]

        return logp, states

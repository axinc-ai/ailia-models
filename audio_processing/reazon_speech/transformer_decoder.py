import numpy as np

onnx = False


def subsequent_mask(size):
    """Create mask for subsequent steps (size, size).

    :param int size: size of mask
    :rtype: numpy.ndarray
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = np.ones((size, size), dtype=bool)
    return np.tril(ret)


class BaseTransformerDecoder(object):
    """Base class of Transfomer decoder module.
    """

    def __init__(
            self,
            decoder,
            num_blocks: int = 6,
    ):
        self.decoder = decoder
        self.num_blocks = num_blocks

    def select_state(self, state, i: int, new_id: int = None):
        """Select state with relative ids in the main beam search.
        Args
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label index to select a state if necessary
        Returns:
            state: pruned state
        """
        return None if state is None else state[i]

    def batch_score(
            self, ys, states, xs):
        """Score new token batch.
        Args:
            ys: torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs:
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        # merge states
        n_batch = len(ys)
        n_layers = self.num_blocks
        if states[0] is None:
            batch_state = [
                np.stack([np.zeros((0, 512), dtype=np.float32) for _ in range(n_batch)])
                for _ in range(n_layers)
            ]
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                np.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = np.expand_dims(subsequent_mask(ys.shape[-1]), axis=0)

        # feedforward
        if not onnx:
            output = self.decoder.predict([ys, ys_mask, xs, *batch_state])
        else:
            output = self.decoder.run(None, {
                'tgt': ys, 'tgt_mask': ys_mask, 'memory': xs,
                'cache1': batch_state[0], 'cache2': batch_state[1], 'cache3': batch_state[2],
                'cache4': batch_state[3], 'cache5': batch_state[4], 'cache6': batch_state[5],
            })
        logp = output[0]
        states = output[1:]

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]

        return logp, state_list


class TransformerDecoder(BaseTransformerDecoder):
    def __init__(
            self,
            decoder,
            num_blocks: int = 6,
    ):
        super().__init__(
            decoder=decoder,
            num_blocks=num_blocks,
        )

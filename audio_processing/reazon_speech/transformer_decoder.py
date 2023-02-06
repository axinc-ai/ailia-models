import numpy as np

onnx = True


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

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
            self,
            decoder,
            # vocab_size: int,
            num_blocks: int = 6,
            # encoder_output_size: int,
            # dropout_rate: float = 0.1,
            # positional_dropout_rate: float = 0.1,
            # input_layer: str = "embed",
            # use_output_layer: bool = True,
            # pos_enc_class=PositionalEncoding,
            # normalize_before: bool = True,
    ):
        self.decoder = decoder
        self.num_blocks = num_blocks

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
        ys = ys[:, -1:].astype(int)
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
            # vocab_size: int,
            # encoder_output_size: int,
            # attention_heads: int = 4,
            # linear_units: int = 2048,
            num_blocks: int = 6,
            # dropout_rate: float = 0.1,
            # positional_dropout_rate: float = 0.1,
            # self_attention_dropout_rate: float = 0.0,
            # src_attention_dropout_rate: float = 0.0,
            # input_layer: str = "embed",
            # use_output_layer: bool = True,
            # pos_enc_class=PositionalEncoding,
            # normalize_before: bool = True,
            # concat_after: bool = False,
            # layer_drop_rate: float = 0.0,
    ):
        super().__init__(
            decoder=decoder,
            # vocab_size=vocab_size,
            num_blocks=num_blocks,
            # encoder_output_size=encoder_output_size,
            # dropout_rate=dropout_rate,
            # positional_dropout_rate=positional_dropout_rate,
            # input_layer=input_layer,
            # use_output_layer=use_output_layer,
            # pos_enc_class=pos_enc_class,
            # normalize_before=normalize_before,
        )

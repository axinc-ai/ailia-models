from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch



class TextTokenCollater:
    """Collate list of text tokens

    Map sentences to integers. Sentences are padded to equal length.
    Beginning and end-of-sequence symbols can be added.

    Example:
        >>> token_collater = TextTokenCollater(text_tokens)
        >>> tokens_batch, tokens_lens = token_collater(text)

    Returns:
        tokens_batch: IntTensor of shape (B, L)
            B: batch dimension, number of input sentences
            L: length of the longest sentence
        tokens_lens: IntTensor of shape (B,)
            Length of each sentence after adding <eos> and <bos>
            but before padding.
    """

    def __init__(
        self,
        pad_symbol: str = "<pad>",
    ):
        self.pad_symbol = pad_symbol

    def __call__(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_seqs = [[p for p in text] for text in texts]
        max_len = len(max(tokens_seqs, key=len))

        seqs = [
            list(seq)
            + [self.pad_symbol] * (max_len - len(seq))
            for seq in tokens_seqs
        ]

        tokens_batch = torch.from_numpy(
            np.array(
                [seq for seq in seqs],
                dtype=np.int64,
            )
        )

        tokens_lens = torch.IntTensor(
            [
                len(seq)
                for seq in tokens_seqs
            ]
        )

        return tokens_batch, tokens_lens


def get_text_token_collater() -> TextTokenCollater:
    collater = TextTokenCollater()
    return collater

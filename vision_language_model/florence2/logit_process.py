from typing import Iterable, List

import numpy as np


def _get_ngrams(ngram_size: int, prev_input_ids: np.ndarray, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []
            ) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: np.ndarray, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(
            generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len
        )
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


def NoRepeatNGramLogitsProcessor(input_ids, scores):
    ngram_size = 3
    num_hypos = scores.shape[0]
    cur_len = input_ids.shape[-1]
    scores_processed = scores.copy()
    banned_batch_tokens = _calc_banned_ngram_tokens(
        ngram_size, input_ids, num_hypos, cur_len
    )
    for i, banned_tokens in enumerate(banned_batch_tokens):
        scores_processed[i, banned_tokens] = -float("inf")

    return scores_processed


def ForcedBOSTokenLogitsProcessor(input_ids, scores):
    bos_token_id = 0
    cur_len = input_ids.shape[-1]
    scores_processed = scores
    if cur_len == 1:
        scores_processed = np.full_like(scores, -float("inf"))
        scores_processed[:, bos_token_id] = 0
    return scores_processed


def ForcedEOSTokenLogitsProcessor(input_ids, scores):
    max_length = 1025
    eos_token_id = 2
    cur_len = input_ids.shape[-1]
    scores_processed = scores
    if cur_len == max_length - 1:
        scores_processed = np.full_like(scores, -float("inf"))
        scores_processed[:, eos_token_id] = 0
    return scores_processed


def logits_processor(input_ids, scores):
    scores = NoRepeatNGramLogitsProcessor(input_ids, scores)
    scores = ForcedBOSTokenLogitsProcessor(input_ids, scores)
    scores = ForcedEOSTokenLogitsProcessor(input_ids, scores)

    return scores

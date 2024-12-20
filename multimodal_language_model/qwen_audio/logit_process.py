from typing import List
import numpy as np

from math_utils import softmax


def StopWordsLogitsProcessor(scores, input_ids):
    eos_token_id = 151643
    stop_words_ids = [[151645], [151644]]

    def tokens_match(prev_tokens: np.ndarray, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        elif len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        elif prev_tokens[-len(tokens) :].tolist() == tokens:
            # if tokens match
            return True
        else:
            return False

    stopped_samples = []
    for prev_input_ids_slice in input_ids:
        match = False
        for stop_token_seq in stop_words_ids:
            if tokens_match(prev_input_ids_slice, stop_token_seq):
                # if tokens do not match continue
                match = True
                break
        stopped_samples.append(match)

    for i, should_stop in enumerate(stopped_samples):
        if should_stop:
            scores[i, eos_token_id] = float(2**15)
    return scores


def TopPLogitsWarper(scores, top_p):
    sorted_indices = np.argsort(scores)
    sorted_logits = np.take_along_axis(scores, sorted_indices, axis=-1)
    cumulative_probs = np.cumsum(softmax(sorted_logits, axis=-1), axis=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    min_tokens_to_keep = 1
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = np.copy(sorted_indices_to_remove)
    np.put_along_axis(
        indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=1
    )

    scores_processed = np.where(indices_to_remove, -np.inf, scores)
    return scores_processed


def logits_processor(input_ids, scores, top_p=0.5):
    scores = StopWordsLogitsProcessor(scores, input_ids)
    scores = TopPLogitsWarper(scores, top_p)

    return scores

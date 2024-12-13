import numpy as np


from math_utils import softmax


def TemperatureLogitsWarper(scores, temperature):
    scores_processed = scores / temperature
    return scores_processed


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
    np.put_along_axis(indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=1)

    scores_processed = np.where(indices_to_remove, -np.inf, scores)
    return scores_processed


def TopKLogitsWarper(scores, top_k):
    # Remove all tokens with a probability less than the last token of the top-k
    top_k_scores = np.sort(scores)[:, -top_k]
    indices_to_remove = scores < top_k_scores
    scores_processed = np.where(indices_to_remove, -np.inf, scores)
    return scores_processed


def logits_processor(input_ids, scores, temperature=0.1, top_k=50):
    scores = TemperatureLogitsWarper(scores, temperature)
    scores = TopKLogitsWarper(scores, top_k)
    # scores = TopPLogitsWarper(scores, top_p)

    return scores

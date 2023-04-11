import numpy as np
from spacy.lang.ja import Japanese


def tokenize(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return np.array([ids])

def pooled_handler(hidden):
    return hidden.mean(dim=1).squeeze()

def select_sentences(body, min_len=40, max_len=600):
    sentences = body.split('。')

    candidates = []
    for s in sentences:
        if min_len < len(s) < max_len:
            candidates.append(s.strip() + '。')
    return candidates


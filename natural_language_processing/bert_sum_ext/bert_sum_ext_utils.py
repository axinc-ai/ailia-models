import numpy as np
from spacy.lang.ja import Japanese


def tokenize(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return np.array([ids])

def pooled_handler(hidden):
    return hidden.mean(dim=1).squeeze()

def select_sentences(body, min_len=40, max_len=600):
    nlp = Japanese()
    nlp.add_pipe('sentencizer')
    doc = nlp(body)

    candidates = []
    for c in doc.sents:
        if min_len < len(c.text.strip()) < max_len:
            candidates.append(c.text.strip())
    return candidates


import os

bad_endings = ['with', 'in', 'on', 'of', 'a', 'at', 'to', 'for', 'an', 'this', 'his', 'her', 'that', 'the']


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        words = []
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                words.append(ix_to_word[str(ix)])
            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            for j in range(len(words)):
                if words[-j - 1] not in bad_endings:
                    flag = -j
                    break
            words = words[:len(words) + flag]
        txt = ' '.join(words)
        out.append(txt.replace('@@ ', ''))

    return out

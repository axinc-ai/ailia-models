import numpy as np
#from scipy.special import log_softmax, logsumexp

from math_utils import softmax

def log_softmax(x, axis=-1):
    c = x.max()
    logsumexp = np.log(np.exp(x - c).sum())
    return x - c - logsumexp

def logsumexp(ns, axis=-1):
    max = np.max(ns)
    ds = ns - max
    sumOfExp = np.exp(ds).sum()
    return max + np.log(sumOfExp)

class MaximumLikelihoodRanker:
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty):
        self.length_penalty = length_penalty

    def rank(self, tokens, sum_logprobs):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]

        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class GreedyDecoder:
    def __init__(self, temperature, eot):
        self.temperature = temperature
        self.eot = eot

    def reset(self):
        pass

    def update(self, tokens, logits, sum_logprobs, rearrange_kv_cache):
        temperature = self.temperature
        if temperature == 0:
            next_tokens = np.argmax(logits, axis=-1)
        else:
            x = logits / temperature
            probs = softmax(x - logsumexp(x, axis=1).reshape(-1, 1), axis=1)
            next_tokens = np.array([np.random.choice(len(p), p=p) for p in probs])

        logprobs = log_softmax(logits, axis=-1)
        current_logprobs = logprobs[np.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = np.concatenate([tokens, next_tokens[:, None]], axis=-1)

        completed = all(tokens[:, -1] == self.eot)

        return tokens, completed

    def finalize(self, tokens, sum_logprobs):
        # make sure each sequence has at least one EOT token at the end
        tokens = np.pad(tokens, [(0, 0), (0, 0), (0, 1)], constant_values=self.eot)
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder:
    def __init__(self, beam_size: int, eot: int, patience=None):
        self.beam_size = beam_size
        self.eot = eot
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(self, tokens, logits, sum_logprobs, rearrange_kv_cache):
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = log_softmax(logits, axis=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()

                indices = np.argsort(-logprobs[idx])
                indices = indices[:self.beam_size + 1]
                values = logprobs[idx][indices]
                for logprob, token in zip(values, indices):
                    new_logprob = (sum_logprobs[idx] + logprob)
                    sequence = tuple(prefix + [token])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = np.array(next_tokens)
        rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens, sum_logprobs):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j]
                    if len(sequences) >= self.beam_size:
                        break

        tokens = [
            [np.array(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        sum_logprobs = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class SuppressBlank:
    def __init__(self, tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits, tokens):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens:
    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits, tokens):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules:
    def __init__(
            self, tokenizer, sample_begin, max_initial_timestamp_index):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits, tokens):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            seq = [t for t in tokens[k, self.sample_begin:].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin:] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

        # apply the `max_initial_timestamp` option
        if tokens.shape[1] == self.sample_begin and self.max_initial_timestamp_index is not None:
            last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
            logits[:, last_allowed + 1:] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = log_softmax(logits, axis=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logsumexp(logprobs[k, self.tokenizer.timestamp_begin:], axis=-1)
            max_text_token_logprob = np.max(logprobs[k, : self.tokenizer.timestamp_begin])
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf

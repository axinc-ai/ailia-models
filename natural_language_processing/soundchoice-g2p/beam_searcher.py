import numpy as np
from scipy.special import log_softmax, logsumexp

batch_size = 1
blank_index = 2
bos_index = 0
eos_index = 1
vocab_size = 43
max_enc_len = 40
minus_inf = -1e20


class AlivedHypotheses:
    """This class handle the data for the hypotheses during the decoding.

    Arguments
    ---------
        alived_seq : np.ndarray
            The sequence of tokens for each hypothesis.
        alived_log_probs : np.ndarray
            The log probabilities of each token for each hypothesis.
        sequence_scores : np.ndarray
            The sum of log probabilities for each hypothesis.
    """

    def __init__(
        self,
        alived_seq,
        alived_log_probs,
        sequence_scores,
    ):
        self.alived_seq = alived_seq
        self.alived_log_probs = alived_log_probs
        self.sequence_scores = sequence_scores


weights = {
    "ctc": 0.5,
    "rnnlm": 0.0,
    "transformerlm": 0.0,
    "kenlm": 0.0,
    "coverage": 5.0,
    "length": 0.0,
}


def coverage_score(inp_tokens, coverage, attn):
    """This method scores the new beams based on the
    Coverage scorer.

    Arguments
    ---------
    inp_tokens : np.ndarray
        The input tensor of the current timestep.
    coverage : No limit
        The scorer states for this timestep.
    attn : np.ndarray
        The attention weight to be used in CoverageScorer or CTCScorer.
    """
    n_bh = attn.shape[0]
    coverage_score.time_step += 1

    if coverage is None:
        coverage = np.zeros_like(attn)

    coverage = coverage + attn

    # Compute coverage penalty and add it to scores
    threshold = 0.5
    tmp = np.ones_like(coverage) * threshold
    penalty = np.where(coverage > tmp, coverage, tmp)
    penalty = np.sum(penalty, axis=-1)

    penalty = penalty - coverage.shape[-1] * threshold
    penalty = np.repeat(
        np.expand_dims(penalty.reshape(n_bh), axis=1), vocab_size, axis=1
    )

    return -1 * penalty / coverage_score.time_step, coverage


coverage_score.time_step = 0


def ctc_score(inp_tokens, states, attn=None):
    """This method if one step of forwarding operation
    for the prefix ctc scorer.

    Arguments
    ---------
    inp_tokens : np.ndarray
        The last chars of prefix label sequences g, where h = g + c.
    states : tuple
        Previous ctc states.
    attn : np.ndarray
        (batch_size * beam_size, max_enc_len), The attention weights.
    """

    n_bh = inp_tokens.shape[0]
    beam_size = n_bh // batch_size
    last_char = inp_tokens
    ctc_score.prefix_length += 1
    num_candidates = vocab_size
    if states is None:
        # r_prev: (L, 2, batch_size * beam_size)
        r_prev = np.ones((max_enc_len, 2, batch_size, beam_size)) * minus_inf

        # Accumulate blank posteriors at each step
        r_prev[:, 1] = np.expand_dims(
            np.cumsum(ctc_score.x[0, :, :, blank_index], 0), axis=2
        )
        r_prev = r_prev.reshape(-1, 2, n_bh)
        psi_prev = np.zeros((n_bh, vocab_size))
    else:
        r_prev, psi_prev = states

    # for full search
    scoring_table = None
    # Inflate x to (2, -1, batch_size * beam_size, num_candidates)
    # It is used to compute forward probs in a batched way
    x_inflate = np.repeat(
        np.expand_dims(ctc_score.x, axis=3), beam_size, axis=3
    ).reshape(2, -1, n_bh, num_candidates)

    # Prepare forward probs
    r = np.ones((max_enc_len, 2, n_bh, num_candidates)) * minus_inf

    # (Alg.2-6)
    if ctc_score.prefix_length == 0:
        r[0, 0] = x_inflate[0, 0]
    # (Alg.2-10): phi = prev_nonblank + prev_blank = r_t-1^nb(g) + r_t-1^b(g)
    r_sum = logsumexp(r_prev, 1)
    phi = np.repeat(np.expand_dims(r_sum, axis=2), num_candidates, axis=2)

    # (Alg.2-10): if last token of prefix g in candidates, phi = prev_b + 0
    for i in range(n_bh):
        phi[:, i, last_char[i]] = r_prev[:, 1, i]

    # Start, end frames for scoring (|g| < |h|).
    # Scoring based on attn peak if ctc_window_size > 0
    start = max(1, ctc_score.prefix_length)
    end = max_enc_len

    # Compute forward prob log(r_t^nb(h)) and log(r_t^b(h)):
    for t in range(start, end):
        # (Alg.2-11): axis=0, p(h|cur step is nonblank) = [p(prev step=y) + phi] * p(c)
        rnb_prev = r[t - 1, 0]
        # (Alg.2-12): axis=1, p(h|cur step is blank) = [p(prev step is blank) + p(prev step is nonblank)] * p(blank)
        rb_prev = r[t - 1, 1]
        r_ = np.stack([rnb_prev, phi[t - 1], rnb_prev, rb_prev]).reshape(
            2, 2, n_bh, num_candidates
        )
        r[t] = logsumexp(r_, 1) + x_inflate[:, t]

    # Compute the predix prob, psi
    psi_init = np.expand_dims(r[start - 1, 0], axis=0)
    # phi is prob at t-1 step, shift one frame and add it to the current prob p(c)
    phix = (
        np.concatenate((np.expand_dims(phi[0], axis=0), phi[:-1]), axis=0)
        + x_inflate[0]
    )
    # (Alg.2-13): psi = psi + phi * p(c)
    psi = logsumexp(np.concatenate((phix[start:end], psi_init), axis=0), axis=0)

    # (Alg.2-3): if c = <eos>, psi = log(r_T^n(g) + r_T^b(g)), where T is the length of max frames
    for i in range(n_bh):
        psi[i, eos_index] = r_sum[ctc_score.last_frame_index[i // beam_size], i]

    if eos_index != blank_index:
        # Exclude blank probs for joint scoring
        psi[:, blank_index] = minus_inf

    return psi - psi_prev, (r, psi, scoring_table)


ctc_score.last_frame_index = None
ctc_score.prefix_length = -1
ctc_score.x = None


def reset_scorer_mem(x, enc_lens):
    ctc_weight = np.load("ctc_fc_weight.npy")
    ctc_bias = np.load("ctc_fc_bias.npy")

    ctc_score.last_frame_index = np.array([enc_lens]) - 1

    logits = x @ ctc_weight.T + ctc_bias

    shape = logits.shape
    x = logits.reshape(shape[0] * shape[1], shape[2])
    x_act = log_softmax(x, axis=-1)
    x_act = x_act.reshape(shape[0], shape[1], shape[2])
    x = x_act

    # length_to_mask
    mask = np.expand_dims(np.arange(enc_lens), axis=0) < enc_lens
    mask = mask.astype(np.int32)
    mask = 1 - mask
    mask = (
        np.broadcast_to(
            np.expand_dims(mask, axis=-1),
            (1, enc_lens, x.shape[-1]),
        )
        == 1
    )
    x = np.where(mask, minus_inf, x)
    x[:, :, 0] = np.where(mask[:, :, 0], 0, x[:, :, 0])

    # axis=0: xnb, nonblank posteriors, axis=1: xb, blank posteriors
    xnb = x.transpose(1, 0, 2)
    xb = np.repeat(np.expand_dims(xnb[:, :, blank_index], axis=2), vocab_size, axis=2)

    # (2, L, batch_size * beam_size, vocab_size)
    ctc_score.x = np.stack([xnb, xb])


class S2SBeamSearcher:

    def __init__(self, net, onnx=False):
        self.min_decode_ratio = 0
        self.max_decode_ratio = 1.0

        self.beam_size = 16
        self.eos_threshold = 10.0
        self.ctc_weight = 0.5
        self.attn_weight = 1.0 - self.ctc_weight

        self.net = net
        self.onnx = onnx

    def _check_full_beams(self, hyps):
        """This method checks whether hyps has been full.

        Arguments
        ---------
            hyps : List
                This list contains batch_size number.
                Each inside list contains a list stores all the hypothesis for this sentence.

        Returns
        -------
            bool
                Whether the hyps has been full.
        """
        hyps_len = [len(lst) for lst in hyps]
        beams_size = [self.beam_size for _ in range(len(hyps_len))]
        return hyps_len == beams_size

    def _check_eos_threshold(self, log_probs):
        """This method checks whether eos log-probabilities exceed threshold.

        Arguments
        ---------
        log_probs : nd.array
            The log-probabilities.

        Returns
        ------
        cond : torch.BoolTensor
            Each element represents whether the eos log-probabilities will be kept.
        """
        max_probs = np.max(log_probs, axis=-1)
        eos_probs = log_probs[:, eos_index]
        cond = eos_probs > (self.eos_threshold * max_probs)
        return cond

    def init_hypotheses(self):
        """This method initializes the AlivedHypotheses object.

        Returns
        -------
        AlivedHypotheses
            The alived hypotheses filled with the initial values.
        """
        sequence_scores = np.ones(self.n_bh) * float("-inf")
        sequence_scores[self.beam_offset, ...] = 0.0
        return AlivedHypotheses(
            alived_seq=np.zeros((self.n_bh, 0), dtype=int),
            alived_log_probs=np.zeros((self.n_bh, 0)),
            sequence_scores=sequence_scores,
        )

    def _attn_weight_step(
        self, inp_tokens, memory, enc_states, enc_lens, attn, log_probs
    ):
        """This method computes a forward_step."""
        hs, c = memory
        hs = np.zeros((4, 16, 512))
        hs = hs.astype(np.float32)
        c = c.astype(np.float32)
        # inp_tokens = inp_tokens.astype(np.int32)
        enc_lens = enc_lens.astype(np.int32)
        if not self.onnx:
            output = self.net.predict([inp_tokens, hs, c, enc_states, enc_lens])
        else:
            output = self.net.run(
                None,
                {
                    "inp_tokens": inp_tokens,
                    "in_hs": hs,
                    "in_c": c,
                    "enc_states": enc_states,
                    "enc_lens": enc_lens,
                },
            )
        log_probs, hs, c, attn = output
        memory = (hs, c)

        log_probs = self.attn_weight * log_probs
        return log_probs, memory, attn

    def _scorer_step(self, inp_tokens, memory, attn, log_probs):
        """This method call the scorers if scorer is not None.

        Arguments
        ---------
        inp_tokens : nd.array
            The input tensor of the current step.
        scorer_memory : No limit
            The memory variables input for this step.
            (ex. RNN hidden states).
        attn : nd.array
            The attention weight.
        log_probs : nd.array
            The log-probabilities of the current step output.

        Returns
        -------
        log_probs : nd.array
            Log-probabilities of the current step output.
        scorer_memory : No limit
            The memory variables generated in this step.
        """
        new_memory = dict()
        score, new_memory["coverage"] = coverage_score(
            inp_tokens, memory["coverage"], attn
        )
        weights = 5.0
        log_probs += score * weights

        # block blank token if CTC is used
        log_probs[:, blank_index] = minus_inf
        score, new_memory["ctc"] = ctc_score(inp_tokens, memory["coverage"], attn)
        weights = 0.5
        log_probs += score * weights

        return log_probs, new_memory

    def _update_reset_memory(self, enc_states, enc_lens):
        """Call reset memory for each module.

        Arguments
        ---------
            enc_states : np.ndarray
                The encoder states to be attended.
            enc_lens : np.ndarray
                The actual length of each enc_states sequence.

        Returns
        -------
            memory : No limit
                The memory variables generated in this step.
            scorer_memory : No limit
                The memory variables generated in this step.
        """
        hs = None
        # self.dec.attn.reset()
        attn_dim = 256
        c = np.zeros((self.n_bh, attn_dim))
        memory = hs, c

        reset_scorer_mem(enc_states, enc_lens)

        scorer_memory = {"coverage": None, "ctc": None}
        return memory, scorer_memory

    def init_beam_search_data(self, enc_states):
        """Initialize the beam search data.

        Arguments
        ---------
            enc_states : np.ndarray
                The encoder states to be attended.
            wav_len : np.ndarray
                The actual length of each enc_states sequence.

        Returns
        -------
            alived_hyps : AlivedHypotheses
                The alived hypotheses.
            inp_tokens : np.ndarray
                The input tensor of the current step.
            log_probs : np.ndarray
                The log-probabilities of the current step output.
            eos_hyps_and_log_probs_scores : list
                Generated hypotheses (the one that haved reached eos) and log probs scores.
            memory : No limit
                The memory variables generated in this step.
            scorer_memory : No limit
                The memory variables generated in this step.
            attn : np.ndarray
                The attention weight.
            prev_attn_peak : np.ndarray
                The previous attention peak place.
            enc_states : np.ndarray
                The encoder states to be attended.
            enc_lens : np.ndarray
                The actual length of each enc_states sequence.
        """
        enc_lens = enc_states.shape[1]

        self.batch_size = enc_states.shape[0]
        self.n_bh = self.batch_size * self.beam_size

        self.n_out = out_features = 43

        memory, scorer_memory = self._update_reset_memory(enc_states, enc_lens)

        # Inflate the enc_states and enc_len by beam_size times
        enc_states = np.tile(enc_states, [self.beam_size, 1, 1])
        enc_lens = np.tile(enc_lens, [self.beam_size])

        # Using bos as the first input
        inp_tokens = np.ones((self.n_bh,), dtype=int) * bos_index

        # The first index of each sentence.
        self.beam_offset = np.arange(self.batch_size) * self.beam_size

        # initialize sequence scores variables.
        sequence_scores = np.ones((self.n_bh,)) * minus_inf

        # keep only the first to make sure no redundancy.
        sequence_scores[self.beam_offset, ...] = 0.0

        # keep the hypothesis that reaches eos and their corresponding score and log_probs.
        eos_hyps_and_log_probs_scores = [[] for _ in range(self.batch_size)]

        self.min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
        self.max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        # Initialize the previous attention peak to zero
        # This variable will be used when using_max_attn_shift=True
        prev_attn_peak = np.zeros(self.n_bh)
        attn = None

        log_probs = np.zeros((self.n_bh, self.n_out))

        alived_hyps = self.init_hypotheses()

        return (
            alived_hyps,
            inp_tokens,
            log_probs,
            eos_hyps_and_log_probs_scores,
            memory,
            scorer_memory,
            attn,
            prev_attn_peak,
            enc_states,
            enc_lens,
        )

    def search_step(
        self,
        alived_hyps,
        inp_tokens,
        log_probs,
        eos_hyps_and_log_probs_scores,
        memory,
        scorer_memory,
        attn,
        prev_attn_peak,
        enc_states,
        enc_lens,
        step,
    ):
        """A search step for the next most likely tokens.

        Arguments
        ---------
            alived_hyps : AlivedHypotheses
                The alived hypotheses.
            inp_tokens : np.ndarray
                The input tensor of the current step.
            log_probs : np.ndarray
                The log-probabilities of the current step output.
            eos_hyps_and_log_probs_scores : list
                Generated hypotheses (the one that haved reached eos) and log probs scores.
            memory : No limit
                The memory variables input for this step.
                (ex. RNN hidden states).
            scorer_memory : No limit
                The memory variables input for this step.
                (ex. RNN hidden states).
            attn : np.ndarray
                The attention weight.
            prev_attn_peak : np.ndarray
                The previous attention peak place.
            enc_states : np.ndarray
                The encoder states to be attended.
            enc_lens : np.ndarray
                The actual length of each enc_states sequence.
            step : int
                The current decoding step.

        Returns
        -------
            alived_hyps : AlivedHypotheses
                The alived hypotheses.
            inp_tokens : np.ndarray
                The input tensor of the current step.
            log_probs : np.ndarray
                The log-probabilities of the current step output.
            eos_hyps_and_log_probs_scores : list
                Generated hypotheses (the one that haved reached eos) and log probs scores.
            memory : No limit
                The memory variables generated in this step.
            scorer_memory : No limit
                The memory variables generated in this step.
            attn : np.ndarray
                The attention weight.
            prev_attn_peak : np.ndarray
                The previous attention peak place.
            scores : np.ndarray
                The scores of the current step output.
        """
        (
            log_probs,
            memory,
            attn,
        ) = self._attn_weight_step(
            inp_tokens,
            memory,
            enc_states,
            enc_lens,
            attn,
            log_probs,
        )

        # Keep the original value
        log_probs_clone = np.copy(log_probs).reshape(self.batch_size, -1)

        # _eos_threshold_step
        cond = self._check_eos_threshold(log_probs)
        log_probs[:, eos_index] = np.where(
            cond,
            log_probs[:, eos_index],
            minus_inf,
        )

        (
            log_probs,
            scorer_memory,
        ) = self._scorer_step(
            inp_tokens,
            scorer_memory,
            attn,
            log_probs,
        )

        (
            scores,
            candidates,
            predecessors,
            inp_tokens,
            alived_hyps,
        ) = self._compute_scores_and_next_inp_tokens(
            alived_hyps,
            log_probs,
            step,
        )

        memory, scorer_memory, prev_attn_peak = self._update_permute_memory(
            memory, scorer_memory, predecessors, candidates, prev_attn_peak
        )

        alived_hyps = self._update_sequences_and_log_probs(
            log_probs_clone,
            inp_tokens,
            predecessors,
            candidates,
            alived_hyps,
        )

        is_eos = self._update_hyps_and_scores_if_eos_token(
            inp_tokens,
            alived_hyps,
            eos_hyps_and_log_probs_scores,
            scores,
        )

        # Block the paths that have reached eos.
        alived_hyps.sequence_scores.masked_fill_(is_eos, float("-inf"))

        return (
            alived_hyps,
            inp_tokens,
            log_probs,
            eos_hyps_and_log_probs_scores,
            memory,
            scorer_memory,
            attn,
            prev_attn_peak,
            scores,
        )

    def forward(self, enc_states):
        (
            alived_hyps,
            inp_tokens,
            log_probs,
            eos_hyps_and_log_probs_scores,
            memory,
            scorer_memory,
            attn,
            prev_attn_peak,
            enc_states,
            enc_lens,
        ) = self.init_beam_search_data(enc_states)

        max_decode_steps = 40
        for step in range(max_decode_steps):
            # terminate condition
            if self._check_full_beams(eos_hyps_and_log_probs_scores):
                break

            (
                alived_hyps,
                inp_tokens,
                log_probs,
                eos_hyps_and_log_probs_scores,
                memory,
                scorer_memory,
                attn,
                prev_attn_peak,
                scores,
            ) = self.search_step(
                alived_hyps,
                inp_tokens,
                log_probs,
                eos_hyps_and_log_probs_scores,
                memory,
                scorer_memory,
                attn,
                prev_attn_peak,
                enc_states,
                enc_lens,
                step,
            )

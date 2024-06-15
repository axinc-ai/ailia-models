import numpy as np


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


def coverage_score(inp_tokens, coverage, candidates, attn):
    """This method scores the new beams based on the
    Coverage scorer.

    Arguments
    ---------
    inp_tokens : np.ndarray
        The input tensor of the current timestep.
    coverage : No limit
        The scorer states for this timestep.
    candidates : np.ndarray
        (batch_size x beam_size, scorer_beam_size).
        The top-k candidates to be scored after the full scorers.
        If None, scorers will score on full vocabulary set.
    attn : np.ndarray
        The attention weight to be used in CoverageScorer or CTCScorer.
    """
    n_bh = attn.shape[0]
    coverage_score.time_step += 1

    if coverage is None:
        coverage = np.zeros_like(attn)

    coverage = coverage + attn
    vocab_size = 43

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


def ctc_score(inp_tokens, states, candidates=None, attn=None):
    """This method if one step of forwarding operation
    for the prefix ctc scorer.

    Arguments
    ---------
    inp_tokens : np.ndarray
        The last chars of prefix label sequences g, where h = g + c.
    states : tuple
        Previous ctc states.
    candidates : np.ndarray
        (batch_size * beam_size, ctc_beam_size), The topk candidates for rescoring.
        If given, performing partial ctc scoring.
    attn : np.ndarray
        (batch_size * beam_size, max_enc_len), The attention weights.
    """

    batch_size = 1
    n_bh = inp_tokens.shape[0]
    beam_size = n_bh // batch_size
    last_char = inp_tokens
    ctc_score.prefix_length += 1
    vocab_size = 43
    num_candidates = vocab_size if candidates is None else candidates.shape[-1]
    if states is None:
        # r_prev: (L, 2, batch_size * beam_size)
        max_enc_len = 40
        r_prev = np.ones((max_enc_len, 2, batch_size, beam_size)) * -1e20

        # Accumulate blank posteriors at each step
        blank_index = 2
        r_prev[:, 1] = np.expand_dims(
            np.cumsum(self.x[0, :, :, blank_index], 0), axis=2
        )
        print(r_prev)
        print(r_prev.shape)
        1 / 0
        r_prev = r_prev.view(-1, 2, n_bh)
        psi_prev = torch.full(
            (n_bh, self.vocab_size),
            0.0,
            device=self.device,
        )
    else:
        r_prev, psi_prev = states

    # for partial search
    if candidates is not None:
        # The first index of each candidate.
        cand_offset = self.batch_index * self.vocab_size
        scoring_table = torch.full(
            (n_bh, self.vocab_size),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        # Assign indices of candidates to their positions in the table
        col_index = torch.arange(n_bh, device=self.device).unsqueeze(1)
        scoring_table[col_index, candidates] = torch.arange(
            num_candidates, device=self.device
        )
        # Select candidates indices for scoring
        scoring_index = (
            candidates + cand_offset.unsqueeze(1).repeat(1, beam_size).view(-1, 1)
        ).view(-1)
        x_inflate = torch.index_select(
            self.x.view(2, -1, self.batch_size * self.vocab_size),
            2,
            scoring_index,
        ).view(2, -1, n_bh, num_candidates)
    # for full search
    else:
        scoring_table = None
        # Inflate x to (2, -1, batch_size * beam_size, num_candidates)
        # It is used to compute forward probs in a batched way
        x_inflate = (
            self.x.unsqueeze(3)
            .repeat(1, 1, 1, beam_size, 1)
            .view(2, -1, n_bh, num_candidates)
        )

    # Prepare forward probs
    r = torch.full(
        (
            self.max_enc_len,
            2,
            n_bh,
            num_candidates,
        ),
        self.minus_inf,
        device=self.device,
    )
    r.fill_(self.minus_inf)

    # (Alg.2-6)
    if self.prefix_length == 0:
        r[0, 0] = x_inflate[0, 0]
    # (Alg.2-10): phi = prev_nonblank + prev_blank = r_t-1^nb(g) + r_t-1^b(g)
    r_sum = torch.logsumexp(r_prev, 1)
    phi = r_sum.unsqueeze(2).repeat(1, 1, num_candidates)

    # (Alg.2-10): if last token of prefix g in candidates, phi = prev_b + 0
    if candidates is not None:
        for i in range(n_bh):
            pos = scoring_table[i, last_char[i]]
            if pos != -1:
                phi[:, i, pos] = r_prev[:, 1, i]
    else:
        for i in range(n_bh):
            phi[:, i, last_char[i]] = r_prev[:, 1, i]

    # Start, end frames for scoring (|g| < |h|).
    # Scoring based on attn peak if ctc_window_size > 0
    if self.ctc_window_size == 0 or attn is None:
        start = max(1, self.prefix_length)
        end = self.max_enc_len
    else:
        _, attn_peak = torch.max(attn, dim=1)
        max_frame = torch.max(attn_peak).item() + self.ctc_window_size
        min_frame = torch.min(attn_peak).item() - self.ctc_window_size
        start = max(max(1, self.prefix_length), int(min_frame))
        end = min(self.max_enc_len, int(max_frame))

    # Compute forward prob log(r_t^nb(h)) and log(r_t^b(h)):
    for t in range(start, end):
        # (Alg.2-11): dim=0, p(h|cur step is nonblank) = [p(prev step=y) + phi] * p(c)
        rnb_prev = r[t - 1, 0]
        # (Alg.2-12): dim=1, p(h|cur step is blank) = [p(prev step is blank) + p(prev step is nonblank)] * p(blank)
        rb_prev = r[t - 1, 1]
        r_ = torch.stack([rnb_prev, phi[t - 1], rnb_prev, rb_prev]).view(
            2, 2, n_bh, num_candidates
        )
        r[t] = torch.logsumexp(r_, 1) + x_inflate[:, t]

    # Compute the predix prob, psi
    psi_init = r[start - 1, 0].unsqueeze(0)
    # phi is prob at t-1 step, shift one frame and add it to the current prob p(c)
    phix = torch.cat((phi[0].unsqueeze(0), phi[:-1]), dim=0) + x_inflate[0]
    # (Alg.2-13): psi = psi + phi * p(c)
    if candidates is not None:
        psi = torch.full(
            (n_bh, self.vocab_size),
            self.minus_inf,
            device=self.device,
        )
        psi_ = torch.logsumexp(torch.cat((phix[start:end], psi_init), dim=0), dim=0)
        # only assign prob to candidates
        for i in range(n_bh):
            psi[i, candidates[i]] = psi_[i]
    else:
        psi = torch.logsumexp(torch.cat((phix[start:end], psi_init), dim=0), dim=0)

    # (Alg.2-3): if c = <eos>, psi = log(r_T^n(g) + r_T^b(g)), where T is the length of max frames
    for i in range(n_bh):
        psi[i, self.eos_index] = r_sum[self.last_frame_index[i // beam_size], i]

    if self.eos_index != self.blank_index:
        # Exclude blank probs for joint scoring
        psi[:, self.blank_index] = self.minus_inf

    return psi - psi_prev, (r, psi, scoring_table)


ctc_score.prefix_length = -1


def reset_scorer_mem():
    pass


class S2SBeamSearcher:

    def __init__(self, net, onnx=False):
        self.bos_index = 0
        self.eos_index = 1
        self.min_decode_ratio = 0
        self.max_decode_ratio = 1.0

        self.beam_size = 16
        self.eos_threshold = 10.0
        self.ctc_weight = 0.5
        self.attn_weight = 1.0 - self.ctc_weight
        self.minus_inf = -1e20

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
        eos_probs = log_probs[:, self.eos_index]
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
        # if self.scorer is not None:
        #     log_probs, scorer_memory = self.scorer.score(
        #         inp_tokens,
        #         scorer_memory,
        #         attn,
        #         log_probs,
        #         self.beam_size,
        #     )
        new_memory = dict()
        score, new_memory["coverage"] = coverage_score(
            inp_tokens, memory["coverage"], None, attn
        )
        weights = 5.0
        log_probs += score * weights

        # block blank token if CTC is used
        blank_index = 2
        minus_inf = -1e20
        log_probs[:, blank_index] = minus_inf
        score, new_memory["coverage"] = ctc_score(
            inp_tokens, memory["coverage"], None, attn
        )

        # score full candidates
        for k, impl in self.full_scorers.items():
            if k == "ctc":
                # block blank token if CTC is used
                log_probs[:, impl.blank_index] = impl.ctc_score.minus_inf

            score, new_memory[k] = impl.score(inp_tokens, memory[k], None, attn)
            log_probs += score * self.weights[k]

        # select candidates from the results of full scorers for partial scorers
        _, candidates = log_probs.topk(
            int(self.beam_size * self.scorer_beam_scale), dim=-1
        )

        # score pruned tokens candidates
        for k, impl in self.partial_scorers.items():
            score, new_memory[k] = impl.score(inp_tokens, memory[k], candidates, attn)
            log_probs += score * self.weights[k]

        return log_probs, new_memory

        return log_probs, scorer_memory

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
        inp_tokens = np.ones((self.n_bh,), dtype=int) * self.bos_index

        # The first index of each sentence.
        self.beam_offset = np.arange(self.batch_size) * self.beam_size

        # initialize sequence scores variables.
        sequence_scores = np.ones((self.n_bh,)) * self.minus_inf

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
        log_probs[:, self.eos_index] = np.where(
            cond,
            log_probs[:, self.eos_index],
            self.minus_inf,
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

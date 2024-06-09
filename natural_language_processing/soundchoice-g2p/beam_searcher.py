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


class S2SBeamSearcher:

    def __init__(self):
        self.bos_index = 0
        self.min_decode_ratio = 0
        self.max_decode_ratio = 1.0

        self.beam_size = 16
        self.attn_weight = 1.0
        self.ctc_weight = 0.0
        self.minus_inf = -1e20

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

    def _update_reset_memory(self):
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
        # memory = self.reset_mem(self.n_bh)
        hs = None
        # self.dec.attn.reset()
        attn_dim = 256
        c = np.zeros((self.n_bh, attn_dim))
        memory = hs, c

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

        memory, scorer_memory = self._update_reset_memory()

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
        log_probs_clone = log_probs.clone().reshape(self.batch_size, -1)

        (
            log_probs,
            prev_attn_peak,
        ) = self._max_attn_shift_step(
            attn,
            prev_attn_peak,
            log_probs,
        )

        log_probs = self._set_eos_minus_inf_step(
            log_probs,
            step,
            self.min_decode_steps,
        )

        log_probs = self._eos_threshold_step(log_probs)

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

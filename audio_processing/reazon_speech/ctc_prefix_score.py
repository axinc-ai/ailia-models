import numpy as np
from scipy.special import logsumexp

onnx = False


class CTCPrefixScore(object):
    """Batch processing of CTCPrefixScore

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probablities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x, xlens, blank, eos, margin=0):
        """Construct CTC prefix scorer
        :param x: input label posterior sequences (B, T, O)
        :param xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        # In the comment lines,
        # we assume T: input_length, B: batch size, W: beam width, O: output dim.
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.shape[0]
        self.input_length = x.shape[1]
        self.odim = x.shape[2]

        # Pad the rest of posteriors in the batch
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Reshape input x
        xn = x.transpose(1, 0, 2)  # (B, T, O) -> (T, B, O)
        xb = np.expand_dims(xn[:, :, self.blank], axis=2)
        xb = np.broadcast_to(xb, xb.shape[:2] + (self.odim,))
        self.x = np.stack([xn, xb])  # (2, T, B, O)
        self.end_frames = np.array(xlens) - 1

        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = np.arange(self.input_length)
        # Base indices for index conversion
        self.idx_bh = None
        self.idx_b = np.arange(self.batch)
        self.idx_bo = np.expand_dims(self.idx_b * self.odim, axis=1)

    def forward(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels
        :param list y: prefix label sequences
        :param tuple state: previous CTC state
        :param pre_scores: scores for pre-selection of hypotheses (BW, O)
        :param att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """
        output_length = len(y[0]) - 1  # ignore sos
        last_ids = [int(yi[-1]) for yi in y]  # last output label ids
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch  # assuming each utterance has the same # of hyps
        self.scoring_num = scoring_ids.shape[-1] if scoring_ids is not None else 0
        # prepare state info
        if state is None:
            r_prev = np.ones((self.input_length, 2, self.batch, n_hyps)) * self.logzero
            r_prev[:, 1] = np.expand_dims(np.cumsum(self.x[0, :, :, self.blank], axis=0), axis=2)
            r_prev = r_prev.reshape(-1, 2, n_bh)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # select input dimensions for scoring
        if self.scoring_num > 0:
            scoring_idmap = np.ones((n_bh, self.odim), dtype=int) * -1
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = np.arange(n_bh).reshape(-1, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = np.arange(snum)
            scoring_idx = (
                    scoring_ids + np.tile(self.idx_bo, (1, n_hyps)).reshape(-1, 1)
            ).reshape(-1)
            x_ = np.take(
                self.x.reshape(2, -1, self.batch * self.odim), scoring_idx, axis=2,
            ).reshape(2, -1, n_bh, snum)
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = np.tile(np.expand_dims(self.x, axis=3), (1, 1, 1, n_hyps, 1)).reshape(2, -1, n_bh, snum)

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = np.ones((self.input_length, 2, n_bh, snum)) * self.logzero
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = logsumexp(r_prev, axis=1)
        log_phi = np.tile(np.expand_dims(r_sum, axis=2), (1, 1, snum))
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = att_w.dot(self.frame_ids)
            f_min = max(int(f_arg.min()), f_min_prev)
            f_max = max(int(f_arg.max()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]
            rr = np.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).reshape(
                2, 2, n_bh, snum
            )
            r[t] = logsumexp(rr, axis=1) + x_[:, t]

        # compute log prefix probabilities log(psi)
        log_phi_x = np.concatenate((np.expand_dims(log_phi[0], axis=0), log_phi[:-1]), axis=0) + x_[0]
        if scoring_ids is not None:
            log_psi = np.ones((n_bh, self.odim)) * self.logzero
            log_psi_ = logsumexp(
                np.concatenate((log_phi_x[start:end], np.expand_dims(r[start - 1, 0], axis=0)), axis=0),
                axis=0,
            )
            for si in range(n_bh):
                log_psi[si, scoring_ids[si]] = log_psi_[si]
        else:
            log_psi = logsumexp(
                np.concatenate((log_phi_x[start:end], np.expand_dims(r[start - 1, 0], axis=0)), axis=0),
                axis=0,
            )

        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)


class CTCPrefixScorer(object):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc, eos: int):
        """Initialize class.
        Args:
            ctc: The CTC implementation.
            eos (int): The end-of-sequence id.
        """
        self.ctc = ctc
        self.eos = eos
        self.impl = None

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.
        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary
        Returns:
            state: pruned state
        """
        if type(state) == tuple:
            r, log_psi, f_min, f_max, scoring_idmap = state
            s = np.array([log_psi[i, new_id]] * log_psi.shape[1])
            if scoring_idmap is not None:
                scoring_idmap = scoring_idmap.astype(int)
                return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
            else:
                return r[:, :, i, new_id], s, f_min, f_max

        return None if state is None else state[i]

    def batch_init_state(self, x):
        """Get an initial state for decoding.
        Args:
            x: The encoded feature tensor
        Returns: initial state
        """
        # feedforward
        x = np.expand_dims(x, axis=0)  # assuming batch_size = 1
        if not onnx:
            output = self.ctc.predict([x])
        else:
            output = self.ctc.run(None, {'hs_pad': x})
        logp = output[0]

        xlen = np.array([logp.shape[1]])
        self.impl = CTCPrefixScore(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.
        Args:
            y: 1D prefix token
            ids: torch.int64 next token to score
            state: decoder state for prefix tokens
            x: 2D encoder feature that generates ys
        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        batch_state = (
            (
                np.stack([s[0] for s in state], axis=2),
                np.stack([s[1] for s in state]),
                state[0][2],
                state[0][3],
            )
            if state[0] is not None
            else None
        )
        return self.impl.forward(y, batch_state, ids)

from logging import getLogger
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np

logger = getLogger(__name__)


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: np.ndarray
    score: Union[float, np.ndarray] = 0
    scores: Dict[str, Union[float, np.ndarray]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class BeamSearch(object):
    """Beam search implementation."""

    def __init__(
            self,
            scorers: Dict[str, Any],
            weights: Dict[str, float],
            beam_size: int,
            vocab_size: int,
            sos: int,
            eos: int,
            token_list: List[str] = None,
            pre_beam_ratio: float = 1.5,
            pre_beam_score_key: str = None,
            hyp_primer: List[int] = None,
    ):
        # set scorers
        self.weights = weights
        self.scorers = scorers
        self.full_scorers = dict()
        self.part_scorers = dict()

        self.full_scorers['decoder'] = self.scorers['decoder']
        self.full_scorers['lm'] = self.scorers['lm']

        # set configurations
        self.sos = sos
        self.eos = eos

        self.token_list = token_list
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size
        self.pre_beam_score_key = pre_beam_score_key
        self.do_pre_beam = (
                self.pre_beam_score_key is not None
                and self.pre_beam_size < self.n_vocab
                and len(self.part_scorers) > 0
        )


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: np.ndarray = np.array([])  # (batch, maxlen)
    score: np.ndarray = np.array([])  # (batch,)
    length: np.ndarray = np.array([])  # (batch,)
    scores: Dict[str, np.ndarray] = dict()  # values: (batch,)
    states: Dict[str, Dict] = dict()

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""

        # pad_sequence
        lens = np.array([h.yseq.shape for h in hyps])
        shape = np.concatenate([[len(hyps), max(lens[:, 0])], lens[0, 1:]])
        yseq = np.ones(shape) * self.eos
        for i, h in enumerate(hyps):
            yseq[i, :h.yseq.shape[0], ...] = h.yseq

        return BatchHypothesis(
            yseq=yseq,
            length=np.array([len(h.yseq) for h in hyps]),
            score=np.array([h.score for h in hyps]),
            scores={k: np.array([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
        )

    def init_hyp(self, x) -> BatchHypothesis:
        """Get an initial hypothesis data.
        Args:
            x: The encoder output feature
        Returns:
            Hypothesis: The initial hypothesis.
        """
        init_states = {'decoder': None, 'ctc': None, 'lm': None}
        init_scores = {'decoder': 0.0, 'ctc': 0.0, 'lm': 0.0}

        primer = [self.sos]
        return self.batchfy([
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                yseq=np.array(primer),
            )
        ])

    def score_full(
            self, hyp: BatchHypothesis, x) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.
        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x: Corresponding input feature
        Returns:
            Tuple[Dict[str, numpy.ndarray], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`
        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)

        return scores, states

    def search(self, running_hyps: BatchHypothesis, x) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.
        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)
        Returns:
            BatchHypothesis: Best sorted hypotheses
        """
        n_batch = len(running_hyps)
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores = np.zeros(
            (n_batch, self.n_vocab)
        )
        scores, states = self.score_full(running_hyps, np.broadcast_to(x, (n_batch,) + x.shape))
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)
        for k in self.part_scorers:
            print("part_scorers>>>", k, self.weights[k])
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        # TODO(karita): do not use list. use batch instead
        # see also https://github.com/espnet/espnet/pull/1402#discussion_r354561029
        # update hyps
        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for (
                full_prev_hyp_id,
                full_new_token_id,
                part_prev_hyp_id,
                part_new_token_id,
        ) in zip(*self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                )
            )
        return self.batchfy(best_hyps)

    def forward(
            self, x, maxlenratio=0.0, minlenratio=0.0):
        maxlen = x.shape[0]
        minlen = int(minlenratio * x.shape[0])

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        ended_hyps = []
        for i in range(maxlen):
            logger.debug("position " + str(i))
            best = self.search(running_hyps, x)
            # post process of one iteration
            running_hyps = self.post_process(i, maxlen, maxlenratio, best, ended_hyps)
            # end detection
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logger.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logger.info("no hypothesis. Finish decoding.")
                break
            else:
                logger.debug(f"remained hypotheses: {len(running_hyps)}")

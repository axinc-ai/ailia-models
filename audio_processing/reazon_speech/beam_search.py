from logging import getLogger
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np
from scipy.special import log_softmax

logger = getLogger(__name__)

onnx = True


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


class ScorerInterface:
    pass


class BeamSearch(object):
    """Beam search implementation."""

    def __init__(
            self,
            scorers: Dict[str, ScorerInterface],
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
        super().__init__()
        # set scorers
        self.weights = weights
        self.scorers = dict()
        self.full_scorers = dict()
        self.part_scorers = dict()

        # for k, v in scorers.items():
        #     w = weights.get(k, 0)
        #     if w == 0 or v is None:
        #         continue
        #     assert isinstance(
        #         v, ScorerInterface
        #     ), f"{k} ({type(v)}) does not implement ScorerInterface"
        #     self.scorers[k] = v
        #     if isinstance(v, PartialScorerInterface):
        #         self.part_scorers[k] = v
        #     else:
        #         self.full_scorers[k] = v
        #     if isinstance(v, torch.nn.Module):
        #         self.nn_dict[k] = v

        # set configurations
        self.sos = sos
        self.eos = eos

        self.token_list = token_list
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size
        if (pre_beam_score_key is not None
                and pre_beam_score_key != "full"
                and pre_beam_score_key not in self.full_scorers):
            raise KeyError(f"{pre_beam_score_key} is not found in {self.full_scorers}")
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
            x (torch.Tensor): Corresponding input feature
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`
        """
        pass

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

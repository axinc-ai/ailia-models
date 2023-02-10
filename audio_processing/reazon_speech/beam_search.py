from logging import getLogger
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy as np
import six

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


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection.
    described in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"
    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in six.moves.range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x["yseq"]) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(
                hyps_same_length, key=lambda x: x["score"], reverse=True
            )[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


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
            pre_beam_score_key: str = None):
        # set scorers
        self.weights = weights
        self.scorers = scorers
        self.full_scorers = dict()
        self.part_scorers = dict()

        self.full_scorers['decoder'] = self.scorers['decoder']
        self.full_scorers['lm'] = self.scorers['lm']
        self.part_scorers['ctc'] = self.scorers['ctc']

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

    def init_hyp(self, x) -> List[Hypothesis]:
        """Get an initial hypothesis data.
        Args:
            x: The encoder output feature
        Returns:
            Hypothesis: The initial hypothesis.
        """
        pass

    @staticmethod
    def append_token(xs: np.ndarray, x: int) -> np.ndarray:
        """Append new token to prefix tokens.
        Args:
            xs (np.ndarray): The prefix token
            x (int): The new token to append
        Returns:
            np.ndarray: New tensor contains: xs + [x] with xs.dtype and xs.device
        """
        x = np.array([x])
        return np.concatenate((xs, x))

    @staticmethod
    def merge_scores(
            prev_scores: Dict[str, float],
            next_full_scores: Dict[str, np.ndarray],
            full_idx: int,
            next_part_scores: Dict[str, np.ndarray],
            part_idx: int) -> Dict[str, np.ndarray]:
        """Merge scores for new hypothesis.
        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_full_scores (Dict[str, np.ndarray]): scores by `self.full_scorers`
            full_idx (int): The next token id for `next_full_scores`
            next_part_scores (Dict[str, np.ndarray]):
                scores of partial tokens by `self.part_scorers`
            part_idx (int): The new token id for `next_part_scores`
        Returns:
            Dict[str, np.ndarray]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.
        """
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores[k] + v[full_idx]
        for k, v in next_part_scores.items():
            new_scores[k] = prev_scores[k] + v[part_idx]
        return new_scores

    def merge_states(self, states, part_states, part_idx: int):
        """Merge states for new hypothesis.
        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`
        Returns:
            Dict[str, np.ndarray]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.
        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, d in self.part_scorers.items():
            new_states[k] = d.select_state(part_states[k], part_idx)
        return new_states

    def search(
            self, running_hyps: List[Hypothesis], x) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.
        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x: Encoded speech feature (T, D)
        Returns:
            List[Hypotheses]: Best sorted hypotheses
        """
        pass

    def forward(
            self, x, maxlenratio=0.0, minlenratio=0.0):
        maxlen = x.shape[0]
        minlen = int(minlenratio * x.shape[0])
        logger.info("decoder input length: " + str(x.shape[0]))
        logger.info("max output length: " + str(maxlen))
        logger.info("min output length: " + str(minlen))

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

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logger.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logger.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logger.info(f"total log probability: {best.score:.2f}")
        logger.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logger.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        # logger.info("best hypo: " + "".join([self.token_list[x] for x in best.yseq[1:-1]]))

        if best.yseq[1:-1].shape[0] == x.shape[0]:
            logger.warning(
                "best hypo length: {} == max output length: {}".format(
                    best.yseq[1:-1].shape[0], maxlen
                )
            )
            logger.warning(
                "decoding may be stopped by the max output length limitation, "
                "please consider to increase the maxlenratio."
            )

        return nbest_hyps

    def post_process(
            self,
            i: int,
            maxlen: int,
            maxlenratio: float,
            running_hyps: List[Hypothesis],
            ended_hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations.
        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (List[Hypothesis]): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.
        Returns:
            List[Hypothesis]: The new running hypotheses.
        """
        pass


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
        yseq = np.ones(shape, dtype=int) * self.eos
        for i, h in enumerate(hyps):
            yseq[i, :h.yseq.shape[0], ...] = h.yseq

        return BatchHypothesis(
            yseq=yseq,
            length=np.array([len(h.yseq) for h in hyps]),
            score=np.array([h.score for h in hyps]),
            scores={k: np.array([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
        )

    def _batch_select(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={
                    k: v.select_state(batch_hyps.states[k], i)
                    for k, v in self.scorers.items()
                },
            )
            for i in range(len(batch_hyps.length))
        ]

    def init_hyp(self, x) -> BatchHypothesis:
        """Get an initial hypothesis data.
        Args:
            x: The encoder output feature
        Returns:
            Hypothesis: The initial hypothesis.
        """
        init_states = {'decoder': None, 'ctc': None, 'lm': None}
        init_scores = {'decoder': 0.0, 'ctc': 0.0, 'lm': 0.0}
        init_states['ctc'] = self.scorers['ctc'].batch_init_state(x)

        primer = [self.sos]
        return self.batchfy([
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                yseq=np.array(primer),
            )
        ])

    def batch_beam(
            self, weighted_scores, ids) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batch-compute topk full token ids and partial token ids.
        Args:
            weighted_scores: The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids: The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`
        """
        top_ids = np.argsort(-weighted_scores.reshape(-1))[:self.beam_size]
        # Because of the flatten above, `top_ids` is organized as:
        # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
        # where V is `self.n_vocab` and K is `self.beam_size`
        prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab

        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

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

    def score_partial(
            self, hyp: BatchHypothesis, ids, x) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.batch_score_partial(
                hyp.yseq, ids, hyp.states[k], x
            )
        return scores, states

    def merge_states(self, states, part_states, part_idx: int):
        """Merge states for new hypothesis.
        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`
        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.
        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, v in part_states.items():
            new_states[k] = v
        return new_states

    def search(self, running_hyps: BatchHypothesis, x) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.
        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (np.ndarray): Encoded speech feature (T, D)
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
            # topk
            part_ids = np.argsort(-pre_beam_scores, axis=-1)
            part_ids = part_ids[..., :self.pre_beam_size]

        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += np.expand_dims(running_hyps.score, axis=1)

        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for full_prev_hyp_id, full_new_token_id, part_prev_hyp_id, part_new_token_id \
                in zip(*self.batch_beam(weighted_scores, part_ids)):
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

    def post_process(
            self,
            i: int,
            maxlen: int,
            maxlenratio: float,
            running_hyps: BatchHypothesis,
            ended_hyps: List[Hypothesis]) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.
        Returns:
            BatchHypothesis: The new running hypotheses.
        """
        n_batch = running_hyps.yseq.shape[0]
        logger.debug(f"the number of running hypothes: {n_batch}")
        if self.token_list is not None:
            logger.debug(
                "best hypo: " + "".join([
                    self.token_list[x]
                    for x in running_hyps.yseq[0, 1: running_hyps.length[0]]
                ])
            )

        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logger.info("adding <eos> in the last position in the loop")
            yseq_eos = np.concatenate(
                (
                    running_hyps.yseq,
                    np.ones((n_batch, 1), dtype=int) * self.eos
                ),
                axis=1,
            )
            running_hyps.yseq.resize(yseq_eos)
            running_hyps.yseq[:] = yseq_eos
            running_hyps.length[:] = yseq_eos.shape[1]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        is_eos = (running_hyps.yseq[np.arange(n_batch), running_hyps.length - 1] == self.eos)
        for b in np.nonzero(is_eos)[0]:
            hyp = self._select(running_hyps, b)
            ended_hyps.append(hyp)
        remained_ids = np.nonzero(is_eos == 0)[0]

        return self._batch_select(running_hyps, remained_ids)

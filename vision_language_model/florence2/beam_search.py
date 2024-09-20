from typing import List, Optional

import numpy as np


class BeamSearchScorer:
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        length_penalty=1.0,
        do_early_stopping=False,
        num_beam_hyps_to_keep=1,
        max_length=None,
    ):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.group_size = self.num_beams

        self._is_init = False

        # self._beam_hyps[i*self.num_beam_groups+j] is the beam_hyps of the j-th group in the i-th mini-batch.
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size)
        ]
        # self._done[i*self.num_beam_groups+j] indicates whether the generation of the beam_hyps of the j-th group
        # in the i-th mini-batch is complete.
        self._done = np.array([False for _ in range(batch_size)], dtype=bool)

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: np.ndarray,
        next_scores: np.ndarray,
        next_tokens: np.ndarray,
        next_indices: np.ndarray,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        decoder_prompt_len=0,
    ) -> List[np.ndarray]:
        # add up to the length which the next_scores is calculated on (including decoder prompt)
        cur_len = input_ids.shape[-1] + 1
        batch_size = len(self._beam_hyps)

        next_beam_scores = np.zeros(
            (batch_size, self.group_size),
        )
        next_beam_tokens = np.zeros(
            (batch_size, self.group_size),
        )
        next_beam_indices = np.zeros(
            (batch_size, self.group_size),
        )

        eos_token_id = [eos_token_id]
        eos_token_id = np.array(eos_token_id)

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx
            if self._done[batch_group_idx]:
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(
                    next_tokens[batch_idx],
                    next_scores[batch_idx],
                    next_indices[batch_idx],
                )
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = (
                        beam_token_rank >= self.group_size
                    )
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_index = None

                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                        generated_len=cur_len - decoder_prompt_len,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_group_idx] = self._done[
                batch_group_idx
            ] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max(), cur_len, decoder_prompt_len
            )

        return {
            "next_beam_scores": next_beam_scores.reshape(-1),
            "next_beam_tokens": next_beam_tokens.reshape(-1),
            "next_beam_indices": next_beam_indices.reshape(-1),
        }

    def finalize(
        self,
        input_ids: np.ndarray,
        final_beam_scores: np.ndarray,
        final_beam_tokens: np.ndarray,
        final_beam_indices: np.ndarray,
        max_length: int,
        # pad_token_id: Optional[Union[int, np.array]] = None,
        # eos_token_id: Optional[Union[int, List[int], np.array]] = None,
        # beam_indices: Optional[np.ndarray] = None,
        # decoder_prompt_len: Optional[int] = 0,
    ) -> List[np.ndarray]:
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if eos_token_id is not None and not isinstance(eos_token_id, np.array):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = np.array(eos_token_id)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_group_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for index_per_group in range(self.group_size):
                batch_beam_idx = batch_group_idx * self.group_size + index_per_group
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = (
                    beam_indices[batch_beam_idx] if beam_indices is not None else None
                )
                generated_len = final_tokens.shape[-1] - decoder_prompt_len
                beam_hyp.add(
                    final_tokens,
                    final_score,
                    beam_indices=beam_index,
                    generated_len=generated_len,
                )

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = np.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            device=self.device,
            dtype=torch.float32,
        )

        # retrieve best hypotheses
        for i in range(batch_size):
            beam_hyps_in_batch = self._beam_hyps[
                i * self.num_beam_groups : (i + 1) * self.num_beam_groups
            ]
            candidate_beams = [
                beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams
            ]
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = (
            min(sent_lengths_max, max_length)
            if max_length is not None
            else sent_lengths_max
        )
        decoded: np.ndarray = input_ids.new(
            batch_size * self.num_beam_hyps_to_keep, sent_max_len
        )

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: np.ndarray = input_ids.new(
                batch_size * self.num_beam_hyps_to_keep, sent_max_len
            )
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = np.array(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


class BeamHypotheses:
    def __init__(
        self,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
        max_length: Optional[int] = None,
    ):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def is_done(
        self,
        best_sum_logprobs: float,
        cur_len: int,
        decoder_prompt_len: int = 0,
    ) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = (
                best_sum_logprobs
                / (cur_len - decoder_prompt_len) ** self.length_penalty
            )
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
            # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                if self.max_length <= decoder_prompt_len:
                    raise ValueError(
                        "max_length is not larger than decoder prompt length"
                    )
                highest_attainable_score = (
                    best_sum_logprobs
                    / (self.max_length - decoder_prompt_len) ** self.length_penalty
                )
            # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
            else:
                highest_attainable_score = (
                    best_sum_logprobs
                    / (cur_len - decoder_prompt_len) ** self.length_penalty
                )
            ret = self.worst_score >= highest_attainable_score
            return ret

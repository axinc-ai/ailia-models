import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Union
from logging import getLogger

import numpy as np
import librosa
from scipy.special import log_softmax

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = 'reazonspeech-nemo-v2_encoder.onnx'
MODEL_ENC_PATH = 'reazonspeech-nemo-v2_encoder.onnx.prototxt'
WEIGHT_DEC_PATH = 'reazonspeech-nemo-v2_decoder.onnx'
MODEL_DEC_PATH = 'reazonspeech-nemo-v2_decoder.onnx.prototxt'
WEIGHT_JNT_PATH = 'reazonspeech-nemo-v2_joint.onnx'
MODEL_JNT_PATH = 'reazonspeech-nemo-v2_joint.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/reason_speech2/'

WAV_PATH = 'speech-001.wav'
SAVE_TEXT_PATH = 'output.txt'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ReazonSpeech2', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

# ======================
# Secondaty Functions
# ======================

vocab_size = 3000
blank_id = 3000
index_incr = 0
beam_size = 4
alsd_max_target_length = 1.0
pred_rnn_layers = 2


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.
    """

    score: float
    y_sequence: Union[List[int], np.ndarray]
    dec_state: Optional[List[np.ndarray]] = None
    timestep: Union[List[int], np.ndarray] = list
    length: Union[int, np.ndarray] = 0
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None


def initialize_state(batch):
    """
    Initialize the state of the RNN layers.

    Returns:
        List of numpy.ndarray, each of shape [L, B, H], where
            L = Number of RNN layers
            B = Batch size
            H = Hidden size of RNN.
    """
    pred_hidden = 640

    state = [
        np.zeros((pred_rnn_layers, batch, pred_hidden), dtype=np.float32),
        np.zeros((pred_rnn_layers, batch, pred_hidden), dtype=np.float32)
    ]
    return state


def batch_initialize_states(batch_states: List[np.ndarray], decoder_states: List[List[np.ndarray]]):
    """
    Create batch of decoder states.

   Args:
       batch_states (list): batch of decoder states
          ([L x (B, H)], [L x (B, H)])

       decoder_states (list of list): list of decoder states
           [B x ([L x (1, H)], [L x (1, H)])]

   Returns:
       batch_states (tuple): batch of decoder states
           ([L x (B, H)], [L x (B, H)])
   """
    # LSTM has 2 states
    new_states = [[] for _ in range(len(decoder_states[0]))]
    for layer in range(pred_rnn_layers):
        for state_id in range(len(decoder_states[0])):
            new_state_for_layer = np.stack([s[state_id][layer] for s in decoder_states])
            new_states[state_id].append(new_state_for_layer)

    for state_id in range(len(decoder_states[0])):
        new_states[state_id] = np.stack([state for state in new_states[state_id]])

    return new_states


def batch_select_state(batch_states, idx):
    """Get decoder state from batch of states, for given id.

    Args:
        batch_states (list): batch of decoder states
            ([L x (B, H)], [L x (B, H)])

        idx (int): index to extract state from batch of states

    Returns:
        (list): decoder states for given id
            ([L x (1, H)], [L x (1, H)])
    """

    if batch_states is not None:
        state_list = []
        for state_id in range(len(batch_states)):
            states = [batch_states[state_id][layer][idx] for layer in range(pred_rnn_layers)]
            state_list.append(states)
        return state_list
    else:
        return None


def batch_score_hypothesis(
        models: dict, hypotheses: List[Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Used for batched beam search algorithms. Similar to score_hypothesis method.

    Args:
        models:
        hypotheses: List of Hypotheses.
        cache: Dict which contains a cache to avoid duplicate computations.
        batch_states: List of np.ndarray which represent the states of the RNN for this batch.
            Each state is of shape [L, B, H]

    Returns:
        Returns a tuple (b_y, b_states, lm_tokens) such that:
        b_y is a np.ndarray of shape [B, 1, H] representing the scores of the last tokens in the Hypotheses.
        b_state is a list of list of RNN states, each of shape [L, B, H].
            Represented as B x List[states].
        lm_token is a list of the final integer tokens of the hypotheses in the batch.
    """
    final_batch = len(hypotheses)

    tokens = []
    process = []
    done = [None for _ in range(final_batch)]

    # For each hypothesis, cache the last token of the sequence and the current states
    for i, hyp in enumerate(hypotheses):
        sequence = tuple(hyp.y_sequence)

        if sequence in cache:
            done[i] = cache[sequence]
        else:
            tokens.append(hyp.y_sequence[-1])
            process.append((sequence, hyp.dec_state))

    if process:
        batch = len(process)

        # convert list of tokens to np.ndarray, then reshape.
        tokens = np.array(tokens).reshape(batch, -1)
        dec_states = initialize_state(tokens.shape[0])  # [L, B, H]
        dec_states = batch_initialize_states(dec_states, [d_state for seq, d_state in process])

        net = models["decoder"]
        if not args.onnx:
            output = net.predict([tokens, dec_states[0], dec_states[1]])
        else:
            output = net.run(None, {'tokens': tokens, 'in_state0': dec_states[0], 'in_state1': dec_states[1]})
        y, out_state0, out_state1 = output

        dec_states = (out_state0, out_state1)

    # Update done states and cache shared by entire batch.
    j = 0
    for i in range(final_batch):
        if done[i] is None:
            # Select sample's state from the batch state list
            new_state = batch_select_state(dec_states, j)

            # Cache [1, H] scores of the current y_j, and its corresponding state
            done[i] = (y[j], new_state)
            cache[process[j][0]] = (y[j], new_state)

            j += 1

    # Set the incoming batch states with the new states obtained from `done`.
    batch_states = batch_initialize_states(batch_states, [d_state for y_j, d_state in done])

    # Create batch of all output scores
    # List[1, 1, H] -> [B, 1, H]
    batch_y = np.stack([y_j for y_j, d_state in done])

    # Extract the last tokens from all hypotheses and convert to a tensor
    lm_tokens = np.array([h.y_sequence[-1] for h in hypotheses]).reshape(final_batch)

    return batch_y, batch_states, lm_tokens


def align_length_sync_decoding(
        models, h, encoded_lengths):
    ids = list(range(vocab_size + 1))
    ids.remove(blank_id)

    # prepare the batched beam states
    beam = min(beam_size, vocab_size)

    h = h[0]  # [T, D]
    h_length = int(encoded_lengths)
    beam_state = initialize_state(
        beam
    )  # [L, B, H], [L, B, H] for LSTMS

    u_max = int(alsd_max_target_length * h_length)

    # Initialize first hypothesis for the beam (blank)
    B = [
        Hypothesis(
            y_sequence=[blank_id],
            score=0.0,
            dec_state=batch_select_state(beam_state, 0),
            timestep=[-1],
            length=0,
        )
    ]

    final = []
    cache = {}

    # ALSD runs for T + U_max steps
    for i in range(h_length + u_max):
        # Update caches
        A = []
        B_ = []
        h_states = []

        # preserve the list of batch indices which are added into the list
        # and those which are removed from the list
        # This is necessary to perform state updates in the correct batch indices later
        batch_ids = list(range(len(B)))  # initialize as a list of all batch ids
        batch_removal_ids = []  # update with sample ids which are removed

        for bid, hyp in enumerate(B):
            u = len(hyp.y_sequence) - 1
            t = i - u

            if t > (h_length - 1):
                batch_removal_ids.append(bid)
                continue

            B_.append(hyp)
            h_states.append((t, h[t]))

        if B_:
            # Compute the subset of batch ids which were *not* removed from the list above
            sub_batch_ids = None
            if len(B_) != beam:
                sub_batch_ids = batch_ids
                for id in batch_removal_ids:
                    # sub_batch_ids contains list of ids *that were not removed*
                    sub_batch_ids.remove(id)

                # extract the states of the sub batch only.
                beam_state_ = [
                    beam_state[state_id][:, sub_batch_ids, :] for state_id in range(len(beam_state))
                ]
            else:
                # If entire batch was used (none were removed), simply take all the states
                beam_state_ = beam_state

            # Decode a batch/sub-batch of beam states and scores
            beam_y, beam_state_, beam_lm_tokens = batch_score_hypothesis(models, B_, cache, beam_state_)

            # If only a subset of batch ids were updated (some were removed)
            if sub_batch_ids is not None:
                # For each state in the RNN (2 for LSTM)
                for state_id in range(len(beam_state)):
                    # Update the current batch states with the sub-batch states (in the correct indices)
                    # These indices are specified by sub_batch_ids, the ids of samples which were updated.
                    beam_state[state_id][:, sub_batch_ids, :] = beam_state_[state_id][...]
            else:
                # If entire batch was updated, simply update all the states
                beam_state = beam_state_

            # h_states = list of [t, h[t]]
            # so h[1] here is a h[t] of shape [D]
            # Simply stack all of the h[t] within the sub_batch/batch (T <= beam)
            h_enc = np.stack([h[1] for h in h_states])  # [T=beam, D]
            h_enc = np.expand_dims(h_enc, axis=1)  # [B=beam, T=1, D]; batch over the beams

            net = models["joint"]
            if not args.onnx:
                output = net.predict([h_enc, beam_y])
            else:
                output = net.run(None, {'f': h_enc, 'g': beam_y})
            res = output[0]

            softmax_temperature = 1.0
            beam_logp = log_softmax(
                res / softmax_temperature, axis=-1
            )  # [B=beam, 1, 1, V + 1]
            beam_logp = beam_logp[:, 0, 0, :]  # [B=beam, V + 1]
            beam_topk = np.argsort(-beam_logp[:, ids], axis=-1)
            beam_topk = beam_topk[:, :beam]

            for j, hyp in enumerate(B_):
                # For all updated samples in the batch, add it as the blank token
                # In this step, we dont add a token but simply update score
                new_hyp = Hypothesis(
                    score=(hyp.score + float(beam_logp[j, blank_id])),
                    y_sequence=hyp.y_sequence[:],
                    dec_state=hyp.dec_state,
                    lm_state=hyp.lm_state,
                    timestep=hyp.timestep[:],
                    length=i,
                )
                # Add blank prediction to A
                A.append(new_hyp)

                # If the prediction "timestep" t has reached the length of the input sequence
                # we can add it to the "finished" hypothesis list.
                if h_states[j][0] == (h_length - 1):
                    final.append(new_hyp)

                # Here, we carefully select the indices of the states that we want to preserve
                # for the next token (non-blank) update.
                if sub_batch_ids is not None:
                    h_states_idx = sub_batch_ids[j]
                else:
                    h_states_idx = j

                # for each current hypothesis j
                # extract the top token score and top token id for the jth hypothesis
                for k in (beam_topk[j] + index_incr):
                    # create new hypothesis and store in A
                    # Note: This loop does *not* include the blank token!
                    logp = beam_logp[:, ids][j][k]
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(logp)),
                        y_sequence=(hyp.y_sequence[:] + [int(k)]),
                        dec_state=batch_select_state(beam_state, h_states_idx),
                        lm_state=hyp.lm_state,
                        timestep=hyp.timestep[:] + [i],
                        length=i,
                    )
                    A.append(new_hyp)

            # Prune and recombine same hypothesis
            # This may cause next beam to be smaller than max beam size
            # Therefore larger beam sizes may be required for better decoding.
            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
            # B = recombine_hypotheses(B)

        # If B_ is empty list, then we may be able to early exit
        elif len(batch_ids) == len(batch_removal_ids):
            # break early
            break

    return 0


# ======================
# Main functions
# ======================


def decode(models, encoder_output, encoded_lengths):
    print(encoder_output, encoder_output.shape)
    print(encoded_lengths)
    encoder_output = encoder_output.transpose(0, 2, 1)  # (B, T, D)

    for batch_idx in range(len(encoder_output)):
        inseq = encoder_output[batch_idx: batch_idx + 1, : encoded_lengths[batch_idx], :]  # [1, T, D]
        logitlen = encoded_lengths[batch_idx]

        # Execute the specific search strategy
        nbest_hyps = align_length_sync_decoding(
            models, inseq, logitlen
        )  # sorted list of hypothesis

        # # Prepare the list of hypotheses
        # nbest_hyps = pack_hypotheses(nbest_hyps)

    results = []

    return results


def predict(models, audio):
    input_signal_length = len(audio)
    input_signal_length = np.array([input_signal_length])
    audio = np.expand_dims(audio, axis=0)
    net = models['encoder']

    # feedforward
    if not args.onnx:
        output = net.predict([audio, input_signal_length])
    else:
        output = net.run(None, {'input_signal': audio, 'input_signal_length': input_signal_length})
    encoded, encoded_length = output

    results = decode(models, encoded, encoded_length)

    return results


def recognize_from_audio(models):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        audio, rate = librosa.load(audio_path, sr=16000)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(models, audio)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(models, audio)

        for res in output:
            logger.info(f"{res[0]}")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_JNT_PATH, MODEL_JNT_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id)
        joint = ailia.Net(MODEL_JNT_PATH, WEIGHT_JNT_PATH, env_id=env_id)
    else:
        import onnxruntime
        encoder = onnxruntime.InferenceSession(WEIGHT_ENC_PATH)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH)
        joint = onnxruntime.InferenceSession(WEIGHT_JNT_PATH)

    models = {
        'encoder': encoder,
        'decoder': decoder,
        'joint': joint,
    }

    recognize_from_audio(models)


if __name__ == '__main__':
    main()

import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Union
from logging import getLogger

import numpy as np
import librosa

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
        np.zeros((pred_rnn_layers, batch, pred_hidden)),
        np.zeros((pred_rnn_layers, batch, pred_hidden))
    ]
    return state


def batch_select_state(batch_states, idx):
    """Get decoder state from batch of states, for given id.

    Args:
        batch_states (list): batch of decoder states
            ([L x (B, H)], [L x (B, H)])

        idx (int): index to extract state from batch of states

    Returns:
        (tuple): decoder states for given id
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


def align_length_sync_decoding(
        h, encoded_lengths):
    vocab_size = 3000
    blank = 3000
    index_incr = 0
    beam_size = 4
    alsd_max_target_length = 1.0

    ids = list(range(vocab_size + 1))
    ids.remove(blank)

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
            y_sequence=[blank],
            score=0.0,
            dec_state=batch_select_state(beam_state, 0),
            timestep=[-1],
            length=0,
        )
    ]

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
            inseq, logitlen
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

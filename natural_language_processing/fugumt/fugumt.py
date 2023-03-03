import sys
import time
from logging import getLogger

import numpy as np
from scipy.special import log_softmax

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

from beam_search import BeamSearchScorer

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_PATH = 'seq2seq-lm-with-past.onnx'
MODEL_PATH = 'seq2seq-lm-with-past.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/fugumt/'

SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'FuguMT', None, SAVE_IMAGE_PATH
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

def logits_processor(input_ids, scores):
    max_length = 512
    eos_token_id = [0]

    cur_len = input_ids.shape[-1]
    if cur_len == max_length - 1:
        num_tokens = scores.shape[1]
        scores[:, [i for i in range(num_tokens) if i not in eos_token_id]] = -float("inf")
        for i in eos_token_id:
            scores[:, i] = 0

    return scores


def reorder_cache(past, beam_idx):
    reordered_past = []
    for i in range(6):
        # cached cross_attention states don't have to be reordered -> they are always the same
        layer_past = past[i * 4:(i + 1) * 4]
        reordered_past.extend((np.take(past_state, beam_idx, axis=0) for past_state in layer_past[:2]))
        reordered_past.extend(layer_past[2:])
    return reordered_past


# ======================
# Main functions
# ======================

def forward(net, input_ids, attention_mask, decoder_input_ids, past_key_values):
    if not args.onnx:
        output = net.predict([
            input_ids, attention_mask, decoder_input_ids,
            past_key_values[0], past_key_values[1], past_key_values[2], past_key_values[3],
            past_key_values[4], past_key_values[5], past_key_values[6], past_key_values[7],
            past_key_values[8], past_key_values[9], past_key_values[10], past_key_values[11],
            past_key_values[12], past_key_values[13], past_key_values[14], past_key_values[15],
            past_key_values[16], past_key_values[17], past_key_values[18], past_key_values[19],
            past_key_values[20], past_key_values[21], past_key_values[22], past_key_values[23],
        ])
    else:
        output = net.run(None, {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'past_key_values.0.decoder.key': past_key_values[0],
            'past_key_values.0.decoder.value': past_key_values[1],
            'past_key_values.0.encoder.key': past_key_values[2],
            'past_key_values.0.encoder.value': past_key_values[3],
            'past_key_values.1.decoder.key': past_key_values[4],
            'past_key_values.1.decoder.value': past_key_values[5],
            'past_key_values.1.encoder.key': past_key_values[6],
            'past_key_values.1.encoder.value': past_key_values[7],
            'past_key_values.2.decoder.key': past_key_values[8],
            'past_key_values.2.decoder.value': past_key_values[9],
            'past_key_values.2.encoder.key': past_key_values[10],
            'past_key_values.2.encoder.value': past_key_values[11],
            'past_key_values.3.decoder.key': past_key_values[12],
            'past_key_values.3.decoder.value': past_key_values[13],
            'past_key_values.3.encoder.key': past_key_values[14],
            'past_key_values.3.encoder.value': past_key_values[15],
            'past_key_values.4.decoder.key': past_key_values[16],
            'past_key_values.4.decoder.value': past_key_values[17],
            'past_key_values.4.encoder.key': past_key_values[18],
            'past_key_values.4.encoder.value': past_key_values[19],
            'past_key_values.5.decoder.key': past_key_values[20],
            'past_key_values.5.decoder.value': past_key_values[21],
            'past_key_values.5.encoder.key': past_key_values[22],
            'past_key_values.5.encoder.value': past_key_values[23],
        })
    logits, *past_key_values, _ = output

    return logits, past_key_values


def beam_search(input_ids, net):
    batch_size = 1
    num_beams = 12
    max_length = 512

    decoder_start_token_id = pad_token_id = 32000
    eos_token_id = 0

    # prepare beam search scorer
    length_penalty = 1.0
    early_stopping = False
    num_return_sequences = 1
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_early_stopping=early_stopping,
        num_beam_hyps_to_keep=num_return_sequences,
    )

    input_ids = np.array([input_ids] * num_beams, dtype=int)
    attention_mask = np.array([[1, 1, 1, 1, 1, 1]] * num_beams, dtype=int)
    decoder_input_ids = np.ones((num_beams, 1), dtype=int) * decoder_start_token_id
    past_key_values = [np.zeros((num_beams, 8, 0, 64), dtype=np.float32)] * 24

    batch_beam_size, cur_len = decoder_input_ids.shape

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = np.zeros((batch_size, num_beams))
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.reshape((batch_size * num_beams))

    i = 0
    while True:
        print("-----------", i)
        i += 1
        logits, past_key_values = forward(
            net, input_ids, attention_mask, decoder_input_ids[:, -1:], past_key_values
        )
        print("logits---", logits)
        print("logits---", logits.shape)
        print()

        next_token_logits = logits[:, -1, :]
        next_token_logits[:, pad_token_id] = float("-inf")
        next_token_scores = log_softmax(next_token_logits, axis=-1)
        next_token_scores_processed = logits_processor(decoder_input_ids, next_token_scores)

        next_token_scores = \
            next_token_scores_processed \
            + np.broadcast_to(beam_scores[:, None], next_token_scores.shape)

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.reshape(batch_size, num_beams * vocab_size)

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
        next_tokens = np.argsort(-next_token_scores, axis=1)[:, :2 * num_beams]
        next_token_scores = np.stack(next_token_scores[i, next_tokens[i]] for i in range(len(next_tokens)))

        next_indices = next_tokens.astype(int) // vocab_size
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            decoder_input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=None,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        decoder_input_ids = np.concatenate([
            decoder_input_ids[beam_idx, :], np.expand_dims(beam_next_tokens, axis=-1)
        ], axis=-1)

        past_key_values = reorder_cache(past_key_values, beam_idx)

        cur_len = cur_len + 1

        if beam_scorer.is_done or decoder_input_ids.shape[-1] > max_length:
            break

    return


def predict(net, input):
    input_ids = [183, 30, 15, 11126, 4, 0]

    beam_search(input_ids, net)
    return


def recognize_from_text(net):
    args.input = ['This is a cat.', ]

    # input audio loop
    for input in args.input:
        logger.info("input: %s" % input)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(net, input)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(net, input)

        for res in output:
            logger.info(f"{res[0]}")

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)

    recognize_from_text(net)


if __name__ == '__main__':
    main()

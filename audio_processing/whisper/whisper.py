import sys
import time

import numpy as np
import tqdm

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from math_utils import softmax
# logger
from logging import getLogger  # noqa

from audio_utils import SAMPLE_RATE, HOP_LENGTH, CHUNK_LENGTH, N_FRAMES
from audio_utils import load_audio, log_mel_spectrogram, pad_or_trim
from tokenizer import get_tokenizer
from decode_utils import BeamSearchDecoder
from decode_utils import SuppressBlank, SuppressTokens, ApplyTimestampRules

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_COND_STAGE_PATH = 'xxx.onnx'
MODEL_COND_STAGE_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/whisper/'

WAV_PATH = 'demo.png'
SAVE_TEXT_PATH = 'output.txt'

n_mels = 80
n_audio_ctx = 1500
n_audio_state = 768
n_audio_head = 12
n_audio_layer = 12
n_vocab = 51865
n_text_ctx = 448
n_text_state = 768
n_text_head = 12
n_text_layer = 12

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Whisper', WAV_PATH, SAVE_TEXT_PATH, input_ftype='audio'
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

def get_initial_tokens(tokenizer, options):
    # sot_sequence = tokenizer.sot_sequence
    sot_sequence = [50258, 50266, 50359]
    sample_len = options.get("sample_len") or n_text_ctx // 2
    n_ctx = n_text_ctx

    tokens = list(sot_sequence)
    prefix = options.get("prefix", None)
    prompt = options.get("prompt", [])

    if prefix:
        prefix_tokens = (
            tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
        )
        if sample_len is not None:
            max_prefix_len = n_ctx // 2 - sample_len
            prefix_tokens = prefix_tokens[-max_prefix_len:]
        tokens = tokens + prefix_tokens

    if prompt:
        prompt_tokens = (
            tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt
        )
        tokens = [tokenizer.sot_prev] + prompt_tokens[-(n_ctx // 2 - 1):] + tokens

    return tuple(tokens)


def get_suppress_tokens(tokenizer, options):
    suppress_tokens = options["suppress_tokens"]

    if isinstance(suppress_tokens, str):
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

    suppress_tokens.extend(
        [tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm]
    )
    if tokenizer.no_speech is not None:
        # no-speech probability is collected separately
        suppress_tokens.append(tokenizer.no_speech)

    return tuple(sorted(set(suppress_tokens)))


# ======================
# Main functions
# ======================

def get_audio_features(enc_net, mel):
    mel = mel.astype('float16')
    if not args.onnx:
        output = enc_net.predict([mel])
    else:
        output = enc_net.run(None, {'mel': mel})
    audio_features = output[0]

    return audio_features


def inference_logits(dec_net, tokens, audio_features, offset):
    offset = np.array(offset)

    print("> decoder", offset, tokens.shape, audio_features.shape)
    if not args.onnx:
        output = dec_net.predict([tokens, audio_features, offset])
    else:
        output = dec_net.run(None, {'x': tokens, 'xa': audio_features, 'inp': offset})
    logits = output[0]
    print("< decoder", logits.shape)

    return logits


decode_options = {}


def decode(enc_net, dec_net, mel, options):
    # decoder = None
    language = options.get("language") or "en"
    language = "Japanese"
    is_multilingual = n_vocab == 51865
    tokenizer = get_tokenizer(is_multilingual, language=language, task='transcribe')

    n_group = options.get("beam_size") or options.get("best_of") or 1
    n_ctx = n_text_ctx
    sample_len = options.get("sample_len") or n_text_ctx // 2

    initial_tokens = get_initial_tokens(tokenizer, options)
    sample_begin = len(initial_tokens)
    sot_index = initial_tokens.index(tokenizer.sot)

    # logit filters: applies various rules to suppress or penalize certain tokens
    logit_filters = []
    if options.get("suppress_blank"):
        logit_filters.append(SuppressBlank(tokenizer, sample_begin))
    if options.get("suppress_tokens"):
        logit_filters.append(SuppressTokens(get_suppress_tokens(tokenizer, options)))
    if not options.get("without_timestamps"):
        precision = CHUNK_LENGTH / n_audio_ctx  # usually 0.02 seconds
        max_initial_timestamp_index = None
        max_initial_timestamp = options.get("max_initial_timestamp")
        if max_initial_timestamp:
            max_initial_timestamp_index = round(max_initial_timestamp / precision)
        logit_filters.append(
            ApplyTimestampRules(tokenizer, sample_begin, max_initial_timestamp_index)
        )

    # decoder: implements how to select the next tokens, given the autoregressive distribution
    if options.get("beam_size") is not None:
        decoder = BeamSearchDecoder(
            options.get("beam_size"), tokenizer.eot, options.get("patience")
        )
    else:
        decoder = GreedyDecoder(options.temperature, tokenizer.eot)

    # decoder.reset()
    n_audio = mel.shape[0]

    audio_features = get_audio_features(enc_net, mel)
    tokens = np.repeat(np.array([initial_tokens]), n_audio, axis=-1)

    # # detect language if requested, overwriting the language token
    # languages, language_probs = detect_language(audio_features, tokens)
    # if self.options.task == "lang_id":
    #     return [
    #         DecodingResult(audio_features=features, language=language, language_probs=probs)
    #         for features, language, probs in zip(audio_features, languages, language_probs)
    #     ]

    # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
    # audio_features = audio_features.repeat_interleave(n_group, dim=0)
    audio_features = np.repeat(audio_features, n_group, axis=0)
    tokens = np.repeat(tokens, n_group, axis=0)

    # sampling loop
    n_batch = tokens.shape[0]
    sum_logprobs = np.zeros(n_batch)
    no_speech_probs = [np.nan] * n_batch
    initial_token_length = len(initial_tokens)
    offset = 0
    for i in range(sample_len):
        if tokens.shape[-1] > initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        logits = inference_logits(dec_net, tokens, audio_features, offset)
        print("logits---1", logits)
        print("logits---1", logits.shape)
        if i >= 2:
            1 / 0
        offset += tokens.shape[-1]  # less 1500

        if i == 0 and tokenizer.no_speech is not None:  # save no_speech_probs
            probs_at_sot = softmax(logits[:, sot_index], axis=-1)
            no_speech_probs = probs_at_sot[:, tokenizer.no_speech].tolist()

        # now we need to consider the logits at the last token only
        logits = logits[:, -1]

        # apply the logit filters, e.g. for suppressing or applying penalty to
        for logit_filter in logit_filters:
            logit_filter.apply(logits, tokens)
        print("logits---2", logits)
        print("logits---2", logits.shape)

        # expand the tokens tensor with the selected next tokens
        tokens, completed = decoder.update(tokens, logits, sum_logprobs)

        if completed or tokens.shape[-1] > n_ctx:
            break

    # reshape the tensors to have (n_audio, n_group) as the first two dimensions
    audio_features = audio_features[:: n_group]
    no_speech_probs = no_speech_probs[:: n_group]
    assert audio_features.shape[0] == len(no_speech_probs) == n_audio

    tokens = tokens.reshape(n_audio, n_group, -1)
    sum_logprobs = sum_logprobs.reshape(n_audio, n_group)

    # get the final candidates for each group, and slice between the first sampled token and EOT
    tokens, sum_logprobs = decoder.finalize(tokens, sum_logprobs)
    tokens: List[List[Tensor]] = [
        [t[self.sample_begin: (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
    ]

    # select the top-ranked sample in each group
    selected = self.sequence_ranker.rank(tokens, sum_logprobs)
    tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
    texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

    sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
    avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

    fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
    if len(set(map(len, fields))) != 1:
        raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

    result = [
        DecodingResult(
            audio_features=features,
            language=language,
            tokens=tokens,
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            temperature=self.options.temperature,
            compression_ratio=compression_ratio(text),
        )
        for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
    ]

    if single:
        result = result[0]

    return result


def decode_with_fallback(enc_net, dec_net, segment):
    # temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
    temperatures = (0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0)
    kwargs = {**decode_options}
    t = temperatures[0]
    if t == 0:
        best_of = kwargs.pop("best_of", None)
    else:
        best_of = kwargs.get("best_of", None)

    options = {"temperature": t, **kwargs}
    results = decode(enc_net, dec_net, segment, options)

    kwargs.pop("beam_size", None)  # no beam search for t > 0
    kwargs.pop("patience", None)  # no patience for t > 0
    kwargs["best_of"] = best_of  # enable best_of for t > 0
    # for t in temperatures[1:]:
    #     needs_fallback = [
    #         compression_ratio_threshold is not None
    #         and result.compression_ratio > compression_ratio_threshold
    #         or logprob_threshold is not None
    #         and result.avg_logprob < logprob_threshold
    #         for result in results
    #     ]
    #     if any(needs_fallback):
    #         options = DecodingOptions(**kwargs, temperature=t)
    #         retries = model.decode(segment[needs_fallback], options)
    #         for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
    #             results[original_index] = retries[retry_index]

    return results


def predict(wav, enc_net, dec_net):
    mel = log_mel_spectrogram(wav)
    mel = np.expand_dims(mel, axis=0)

    # language = "Japanese"
    # task = "transcribe"
    # tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    seek = 0
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    decode_options.update({
        'task': 'transcribe', 'language': 'Japanese', 'best_of': 5, 'beam_size': 5, 'patience': None,
        'length_penalty': None, 'suppress_tokens': '-1', 'fp16': True, 'prompt': []
    })

    num_frames = mel.shape[-1]
    previous_seek_value = seek
    with tqdm.tqdm(total=num_frames, unit='frames') as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, :, seek:], N_FRAMES)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result = decode_with_fallback(enc_net, dec_net, segment)
            result = result[0]
            tokens = np.array(result.tokens)

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

            break

    return


def recognize_from_audio(enc_net, dec_net):
    # input audio loop
    for audio_path in args.input:
        logger.info(audio_path)

        # prepare input data
        wav = load_audio(audio_path)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(wav, enc_net, dec_net)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            output = predict(wav, enc_net, dec_net)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.txt')
        logger.info(f'saved at : {savepath}')

    logger.info('Script finished successfully.')


def main():
    WEIGHT_ENC_SMALL_PATH = "encoder_small.onnx"
    MODEL_ENC_SMALL_PATH = "encoder_small.onnx.prototxt"
    WEIGHT_DEC_SMALL_PATH = "decoder_small.onnx"
    MODEL_DEC_SMALL_PATH = "decoder_small.onnx.prototxt"
    check_and_download_models(WEIGHT_ENC_SMALL_PATH, MODEL_ENC_SMALL_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_SMALL_PATH, MODEL_DEC_SMALL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        enc_net = ailia.Net(MODEL_ENC_SMALL_PATH, WEIGHT_ENC_SMALL_PATH, env_id=env_id)
        dec_net = ailia.Net(MODEL_DEC_SMALL_PATH, WEIGHT_DEC_SMALL_PATH, env_id=env_id)
    else:
        import onnxruntime
        enc_net = onnxruntime.InferenceSession(WEIGHT_ENC_SMALL_PATH)
        dec_net = onnxruntime.InferenceSession(WEIGHT_DEC_SMALL_PATH)

    recognize_from_audio(enc_net, dec_net)


if __name__ == '__main__':
    main()

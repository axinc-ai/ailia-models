import sys
import time
from collections import namedtuple

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
from decode_utils import MaximumLikelihoodRanker, BeamSearchDecoder
from decode_utils import SuppressBlank, SuppressTokens, ApplyTimestampRules

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_COND_STAGE_PATH = 'xxx.onnx'
MODEL_COND_STAGE_PATH = 'xxx.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/whisper/'

WAV_PATH = 'demo.wav'
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
    '-m', '--model_type', default='small', choices=('small',),
    help='model type'
)
parser.add_argument(
    "--beam_size", type=int, default=5,
    help="number of beams in beam search, only applicable when temperature is zero"
)
parser.add_argument(
    "--logprob_threshold", type=float, default=-1.0,
    help="if the average log probability is lower than this value, treat the decoding as failed"
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

def is_multilingual():
    return n_vocab == 51865


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""

    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


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


def new_kv_cache(n_group, length):
    model_type = args.model_type
    if model_type == "tiny.en" or model_type == "tiny":
        size = [8, n_group, length, 384]
    elif model_type == "base.en" or model_type == "base":
        size = [12, n_group, length, 512]
    elif model_type == "small.en" or model_type == "small":
        size = [24, n_group, length, 768]
    elif model_type == "medium.en" or model_type == "medium":
        size = [48, n_group, length, 1024]
    elif model_type == "large":
        size = [64, n_group, length, 1280]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return np.zeros(size, dtype=np.float32)


# ======================
# Main functions
# ======================

def get_audio_features(enc_net, mel):
    if not args.onnx:
        output = enc_net.predict([mel])
    else:
        output = enc_net.run(None, {'mel': mel})
    audio_features = output[0]

    return audio_features


def inference_logits(dec_net, tokens, audio_features, kv_cache, initial_token_length):
    n_group = tokens.shape[0]
    if kv_cache is None:
        kv_cache = new_kv_cache(n_group, initial_token_length)
        offset = 0
    else:
        offset = kv_cache.shape[2]
        _kv_cache = new_kv_cache(n_group, offset + 1)
        _kv_cache[:, :, :-1, :] = kv_cache
        kv_cache = _kv_cache

    if tokens.shape[-1] > initial_token_length:
        # only need to use the last token except in the first forward pass
        tokens = tokens[:, -1:]

    tokens = tokens.astype(np.int64)
    offset = np.array(offset, dtype=np.int64)

    if not args.onnx:
        output = dec_net.predict([tokens, audio_features, kv_cache, offset])
    else:
        kv_cache = kv_cache.astype(np.float32)
        output = dec_net.run(None, {
            'tokens': tokens, 'audio_features': audio_features,
            'kv_cache': kv_cache, 'offset': offset})
    logits, kv_cache = output

    return logits, kv_cache


def decode(enc_net, dec_net, mel, options):
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    language = options.get("language") or "en"
    language = "Japanese"
    tokenizer = get_tokenizer(is_multilingual(), language=language, task='transcribe')

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

    # sequence ranker: implements how to rank a group of sampled sequences
    sequence_ranker = MaximumLikelihoodRanker(options.get("length_penalty"))

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
    languages = ['Japanese']

    # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
    audio_features = np.repeat(audio_features, n_group, axis=0)
    tokens = np.repeat(tokens, n_group, axis=0)

    n_batch = tokens.shape[0]
    sum_logprobs = np.zeros(n_batch)
    no_speech_probs = [np.nan] * n_batch
    initial_token_length = len(initial_tokens)
    kv_cache = None

    # sampling loop
    for i in range(sample_len):
        print(f"step: {i}", flush=True)
        logits, kv_cache = inference_logits(dec_net, tokens, audio_features, kv_cache, initial_token_length)
        # print("logits---1", logits)
        # print("logits---1", logits.shape)
        # if i >= 10:
        #     1 / 0

        if i == 0 and tokenizer.no_speech is not None:  # save no_speech_probs
            probs_at_sot = softmax(logits[:, sot_index], axis=-1)
            no_speech_probs = probs_at_sot[:, tokenizer.no_speech].tolist()

        # now we need to consider the logits at the last token only
        logits = logits[:, -1]

        # apply the logit filters, e.g. for suppressing or applying penalty to
        for logit_filter in logit_filters:
            logit_filter.apply(logits, tokens)

        # print("logits---2", logits)
        # print("logits---2", logits.shape)

        def rearrange_kv_cache(source_indices):
            kv_cache[...] = kv_cache[:, source_indices]

        # expand the tokens tensor with the selected next tokens
        tokens, completed = decoder.update(tokens, logits, sum_logprobs, rearrange_kv_cache)

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
    tokens = [[
        t[sample_begin: np.nonzero(t == tokenizer.eot)[0][0]] for t in s
    ] for s in tokens]

    # select the top-ranked sample in each group
    selected = sequence_ranker.rank(tokens, sum_logprobs)
    tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
    texts = [tokenizer.decode(t).strip() for t in tokens]

    sum_logprobs = [lp[i] for i, lp in zip(selected, sum_logprobs)]
    avg_logprobs = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

    fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
    if len(set(map(len, fields))) != 1:
        raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

    DecodingResult = namedtuple('DecodingResult', [
        'audio_features', 'language', 'language_probs',
        'tokens', 'text', 'avg_logprob', 'no_speech_prob',
        'temperature',
    ])

    result = [
        DecodingResult(
            audio_features=features,
            language=language,
            language_probs=None,
            tokens=tokens,
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            temperature=options.get("temperature"),
        ) for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
    ]

    if single:
        result = result[0]

    return result


def decode_with_fallback(enc_net, dec_net, segment, decode_options):
    logprob_threshold = args.logprob_threshold

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
    for t in temperatures[1:]:
        needs_fallback = [
            result.avg_logprob < logprob_threshold for result in results
        ]
        if any(needs_fallback):
            options = {"temperature": t, **kwargs}
            retries = decode(enc_net, dec_net, segment[needs_fallback], options)
            for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                results[original_index] = retries[retry_index]

    return results


def predict(wav, enc_net, dec_net):
    no_speech_threshold = 0.6
    logprob_threshold = args.logprob_threshold

    decode_options = {
        'task': 'transcribe', 'language': 'Japanese', 'best_of': 5,
        'beam_size': args.beam_size, 'patience': None,
        'length_penalty': None, 'suppress_tokens': '-1', 'fp16': True, 'prompt': []
    }

    mel = log_mel_spectrogram(wav)
    mel = np.expand_dims(mel, axis=0)
    language = "Japanese"
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(is_multilingual(), language=language, task=task)

    seek = 0
    input_stride = N_FRAMES // n_audio_ctx  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def add_segment(
            start, end, text_tokens, result):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append({
            "id": len(all_segments),
            "seek": seek,
            "start": start,
            "end": end,
            "text": text,
            "tokens": result.tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            # "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        })
        logger.info(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")

    num_frames = mel.shape[-1]
    previous_seek_value = seek

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    with tqdm.tqdm(total=num_frames, unit='frames', disable=True) as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, :, seek:], N_FRAMES)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result = decode_with_fallback(enc_net, dec_net, segment, decode_options)
            result = result[0]
            tokens = np.array(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens = tokens >= tokenizer.timestamp_begin
            consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                            sliced_tokens[0] - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                            sliced_tokens[-1] - tokenizer.timestamp_begin
                    )
                    add_segment(
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=timestamp_offset + end_timestamp_position * time_precision,
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                        tokens[last_slice - 1] - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[np.nonzero(timestamp_tokens)[0]]
                if len(timestamps) > 0:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                )
                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

    d = dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt):]),
        segments=all_segments,
        language=language
    )
    return d


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

        print(output)

        # # output result
        # savepath = get_savepath(args.savepath, audio_path, ext='.txt')
        # logger.info(f'saved at : {savepath}')

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

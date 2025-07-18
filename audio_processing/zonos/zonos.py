import os
import sys
import time
from logging import getLogger

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from math_utils import softmax

from conditioning import supported_language_codes, phonemize, tokenize_phonemes

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_EMB_PATH = "speaker_embedding.onnx"
WEIGHT_PHONEME_PATH = "phoneme_embedder.onnx"
WEIGHT_COND_PATH = "conditioner.onnx"
WEIGHT_FIRST_PATH = "generator_first.onnx"
WEIGHT_STAGE_PATH = "generator_stage.onnx"
WEIGHT_DEC_PATH = "autoencoder.onnx"
MODEL_EMB_PATH = "speaker_embedding.onnx.prototxt"
MODEL_PHONEME_PATH = "phoneme_embedder.onnx.prototxt"
MODEL_COND_PATH = "conditioner.onnx.prototxt"
MODEL_FIRST_PATH = "generator_first.onnx.prototxt"
MODEL_STAGE_PATH = "generator_stage.onnx.prototxt"
MODEL_DEC_PATH = "autoencoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/zonos/"

REF_WAV_PATH = "exampleaudio.mp3"
SAVE_WAV_PATH = "output.wav"


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Zonos", None, SAVE_WAV_PATH)
# overwrite
parser.add_argument(
    "--input",
    "-i",
    metavar="TEXT",
    default="こんにちは",
    help="input text",
)
parser.add_argument(
    "--text_language",
    "-tl",
    default="ja",
    help="input text language. example: en-us, ja,...",
)
parser.add_argument(
    "--ref_audio",
    "-ra",
    metavar="TEXT",
    default=REF_WAV_PATH,
    help="ref audio",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================


class EspeakPhonemeConditioner:
    def __init__(self, phoneme_embedder, flg_onnx=False):
        self.phoneme_embedder = phoneme_embedder
        self.flg_onnx = flg_onnx

    def forward(self, texts: list[str], languages: list[str]):
        """
        Args:
            texts: list of texts to convert to phonemes
            languages: ISO 639-1 -or otherwise eSpeak compatible- language code
        """
        phonemes = phonemize(texts, languages)
        phoneme_ids, _ = tokenize_phonemes(phonemes)
        if not self.flg_onnx:
            output = self.phoneme_embedder.run(
                {"input": phoneme_ids},
            )
        else:
            output = self.phoneme_embedder.run(
                None,
                {"input": phoneme_ids},
            )
        cond = output[0]

        return cond


def prefix_conditioner(net, cond_dict):
    if not args.onnx:
        output = net.run(
            {
                "espeak": cond_dict["espeak"],
                "speaker": cond_dict["speaker"],
                "emotion": cond_dict["emotion"],
                "fmax": cond_dict["fmax"],
                "pitch_std": cond_dict["pitch_std"],
                "speaking_rate": cond_dict["speaking_rate"],
                "language_id": cond_dict["language_id"],
            },
        )
    else:
        output = net.run(
            None,
            {
                "espeak": cond_dict["espeak"],
                "speaker": cond_dict["speaker"],
                "emotion": cond_dict["emotion"],
                "fmax": cond_dict["fmax"],
                "pitch_std": cond_dict["pitch_std"],
                "speaking_rate": cond_dict["speaking_rate"],
                "language_id": cond_dict["language_id"],
            },
        )
    conditioning = output[0]

    return conditioning


def modify_logit_for_repetition_penalty(
    logits: np.ndarray,  # shape: (B, C, V)
    generated_tokens: np.ndarray,  # shape: (B, C, T)
    repetition_penalty: float,
    repetition_penalty_window: int,
):
    """
    NumPy version of repetition penalty logic.
    logits: (batch_size, n_codebooks, vocab_size)
    generated_tokens: (batch_size, n_codebooks, seq_len)
    """
    B, C, V = logits.shape
    T = generated_tokens.shape[-1]

    tokens = generated_tokens[..., -repetition_penalty_window:]
    tokens = np.minimum(tokens, V - 1).astype(np.int64)

    factors = np.ones_like(logits)
    for b in range(B):
        for c in range(C):
            for t in range(tokens.shape[2]):
                idx = tokens[b, c, t]
                factors[b, c, idx] *= repetition_penalty

    logits = np.where(logits <= 0, logits * factors, logits / factors)

    return logits


def sample(
    logits,
    temperature=1.0,
    min_p=0.1,
    generated_tokens: np.ndarray | None = None,
    repetition_penalty: float = 3.0,
    repetition_penalty_window: int = 2,
):
    if repetition_penalty != 1.0 and generated_tokens is not None:
        logits = modify_logit_for_repetition_penalty(
            logits, generated_tokens, repetition_penalty, repetition_penalty_window
        )

    probs = softmax(logits / temperature, axis=-1)

    # apply_min_p
    top_probs = np.max(probs, axis=-1, keepdims=True)
    tokens_to_remove = probs < (min_p * top_probs)
    probs = np.where(tokens_to_remove, 0.0, probs)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    # multinomial
    batch_size, num_heads, vocab_size = probs.shape
    next_token = np.zeros((batch_size, num_heads, 1), dtype=np.int64)
    for b in range(batch_size):
        for h in range(num_heads):
            next_token[b, h, 0] = np.random.choice(vocab_size, p=probs[b, h])

    return next_token


# ======================
# Main functions
# ======================


def generate(models, prefix_conditioning):
    eos_token_id = 1024
    masked_token_id = 1025
    batch_size = 1

    max_new_tokens = 86 * 30
    prefix_audio_len = 0
    audio_seq_len = prefix_audio_len + max_new_tokens
    unknown_token = -1
    codes = np.full((batch_size, 9, audio_seq_len), unknown_token)

    # apply_delay_pattern
    pad_width = ((0, 0), (0, 0), (0, codes.shape[1]))
    codes_padded = np.pad(
        codes, pad_width, mode="constant", constant_values=masked_token_id
    )
    delayed_codes = np.stack(
        [
            np.roll(codes_padded[:, k, :], shift=k + 1, axis=-1)
            for k in range(codes.shape[1])
        ],
        axis=1,
    )
    delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

    net = models["first_net"]
    if not args.onnx:
        output = net.run({"conditioning": prefix_conditioning})
    else:
        output = net.run(None, {"conditioning": prefix_conditioning})
    logits, freqs_cis, *kv_cache = output

    next_token = sample(logits)

    offset = delayed_prefix_audio_codes.shape[2]
    frame = delayed_codes[..., offset : offset + 1]
    mask = frame == unknown_token
    frame[mask] = next_token.flatten()[: np.count_nonzero(mask)]

    prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
    seqlen_offset = np.array(prefix_length)

    logit_bias = np.zeros_like(logits)
    logit_bias[:, 1:, eos_token_id] = -np.inf  # only allow codebook 0 to predict EOS

    stopping = np.zeros(batch_size, dtype=bool)
    max_steps = delayed_codes.shape[2] - offset
    remaining_steps = np.full((batch_size,), max_steps)
    progress = tqdm(total=max_steps, desc="Generating", disable=False)

    step = 0
    while np.max(remaining_steps) > 0:
        offset += 1
        input_ids = delayed_codes[..., offset - 1 : offset]

        kv_cache = {"kv_cache_%d" % i: kv for i, kv in enumerate(kv_cache)}
        net = models["stage_net"]
        if not args.onnx:
            output = net.run(
                {
                    "input_ids": input_ids,
                    "freqs_cis": freqs_cis,
                    "seqlen_offset": seqlen_offset,
                    **kv_cache,
                }
            )
        else:
            output = net.run(
                None,
                {
                    "input_ids": input_ids,
                    "freqs_cis": freqs_cis,
                    "seqlen_offset": seqlen_offset,
                    **kv_cache,
                },
            )
        logits, freqs_cis, *kv_cache = output
        logits += logit_bias

        next_token = sample(logits, generated_tokens=delayed_codes[..., :offset])
        eos_in_cb0 = next_token[:, 0] == eos_token_id

        indices = eos_in_cb0[:, 0]
        remaining_steps[indices] = np.minimum(remaining_steps[indices], 9)
        stopping |= eos_in_cb0[:, 0]

        eos_codebook_idx = 9 - remaining_steps
        eos_codebook_idx = np.clip(eos_codebook_idx, None, a_max=9 - 1)
        for i in range(next_token.shape[0]):
            if stopping[i]:
                idx = eos_codebook_idx[i]
                next_token[i, :idx] = masked_token_id
                next_token[i, idx] = eos_token_id

        frame = delayed_codes[..., offset : offset + 1]
        mask = frame == unknown_token
        frame[mask] = next_token.flatten()[: np.count_nonzero(mask)]
        seqlen_offset += 1

        remaining_steps -= 1
        progress.update()
        step += 1

    # revert_delay_pattern
    _, n_q, seq_len = delayed_codes.shape
    codes = np.stack(
        [delayed_codes[:, k, k + 1 : seq_len - n_q + k + 1] for k in range(n_q)], axis=1
    )
    codes[codes >= 1024] = 0
    codes = codes[..., : offset - 9]

    net = models["decoder"]
    if not args.onnx:
        output = net.run({"codes": codes})
    else:
        output = net.run(None, {"codes": codes})
    wavs = output[0]

    return wavs


def generate_voice(models):
    ref_audio = args.ref_audio
    text = args.input
    text_language = args.text_language

    if not os.path.isfile(ref_audio):
        logger.error("specified input is not file path nor directory path")
        sys.exit(0)

    language_code_to_id = {lang: i for i, lang in enumerate(supported_language_codes)}
    if text_language not in language_code_to_id:
        raise ValueError(
            f"Language '{text_language}' is not supported. "
            f"Supported languages are: {supported_language_codes}"
        )

    logger.info("Actual Input Reference wav: %s" % ref_audio)
    logger.info("Actual Input Target Text: %s (%s)" % (text, text_language))

    wav, _ = librosa.load(ref_audio, sr=16_000)
    wav = wav[None, ...]

    language_id = language_code_to_id[text_language]

    net = models["embedding"]
    if not args.onnx:
        output = net.run({"wav": wav})
    else:
        output = net.run(None, {"wav": wav})
    speaker = output[0]

    net = models["phoneme_embedder"]
    espeak_cond = EspeakPhonemeConditioner(net, args.onnx).forward(
        [text], [text_language]
    )

    # prepare_conditioning
    cond_dict = dict(
        espeak=espeak_cond,
        speaker=speaker,
        emotion=np.array(
            [0.3078, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2565, 0.3078],
            dtype=np.float32,
        ).reshape(1, 1, -1),
        fmax=np.array([22050], dtype=np.float32).reshape(1, 1, -1),
        pitch_std=np.array([20], dtype=np.float32).reshape(1, 1, -1),
        speaking_rate=np.array([15], dtype=np.float32).reshape(1, 1, -1),
        language_id=np.array([language_id], dtype=int).reshape(1, 1, -1),
    )
    uncond_dict = dict(
        espeak=espeak_cond,
        speaker=speaker[:, :0, :],
        emotion=np.zeros((1, 0, 8), dtype=np.float32),
        fmax=np.zeros((1, 0, 1), dtype=np.float32),
        pitch_std=np.zeros((1, 0, 1), dtype=np.float32),
        speaking_rate=np.zeros((1, 0, 1), dtype=np.float32),
        language_id=np.zeros((1, 0, 1), dtype=int),
    )

    net = models["conditioner"]
    conditioning = np.concatenate(
        [prefix_conditioner(net, cond_dict), prefix_conditioner(net, uncond_dict)]
    )

    wavs = generate(models, conditioning)

    # save result
    savepath = get_savepath(args.savepath, ref_audio, ext=".wav")
    sf.write(savepath, wavs[0].T, 44100)
    logger.info(f"saved at : {savepath}")

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_EMB_PATH, MODEL_EMB_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_PHONEME_PATH, MODEL_PHONEME_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_COND_PATH, MODEL_COND_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_FIRST_PATH, MODEL_FIRST_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_STAGE_PATH, MODEL_STAGE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        embedding = ailia.Net(MODEL_EMB_PATH, WEIGHT_EMB_PATH, env_id=env_id)
        phoneme_embedder = ailia.Net(
            MODEL_PHONEME_PATH, WEIGHT_PHONEME_PATH, env_id=env_id
        )
        conditioner = ailia.Net(MODEL_COND_PATH, WEIGHT_COND_PATH, env_id=env_id)
        first_net = ailia.Net(MODEL_FIRST_PATH, WEIGHT_FIRST_PATH, env_id=env_id)
        stage_net = ailia.Net(MODEL_STAGE_PATH, WEIGHT_STAGE_PATH, env_id=env_id)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        embedding = onnxruntime.InferenceSession(WEIGHT_EMB_PATH, providers=providers)
        phoneme_embedder = onnxruntime.InferenceSession(
            WEIGHT_PHONEME_PATH, providers=providers
        )
        conditioner = onnxruntime.InferenceSession(
            WEIGHT_COND_PATH, providers=providers
        )
        first_net = onnxruntime.InferenceSession(WEIGHT_FIRST_PATH, providers=providers)
        stage_net = onnxruntime.InferenceSession(WEIGHT_STAGE_PATH, providers=providers)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    models = {
        "embedding": embedding,
        "phoneme_embedder": phoneme_embedder,
        "conditioner": conditioner,
        "first_net": first_net,
        "stage_net": stage_net,
        "decoder": decoder,
    }

    generate_voice(models)


if __name__ == "__main__":
    main()

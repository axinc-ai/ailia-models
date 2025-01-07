import sys
import time

# logger
from logging import getLogger  # noqa

import librosa
import numpy as np
import whisper

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

import torchaudio.compliance.kaldi as kaldi
from audio_utils import mel_spectrogram
from cosyvoice_utils import text_normalize


logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/cosyvoice/"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("CosyVoice", None, None)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    "--disable_ailia_tokenizer", action="store_true", help="disable ailia tokenizer."
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================


# ======================
# Main functions
# ======================


def extract_speech_token(speech_tokenizer, speech):
    import torch

    assert (
        speech.shape[1] / 16000 <= 30
    ), "do not support extract speech token for audio longer than 30s"
    feat = whisper.log_mel_spectrogram(torch.from_numpy(speech), n_mels=128)
    feat = feat.numpy()

    if not args.onnx:
        output = speech_tokenizer.predict([feat, np.array([feat.shape[2]], dtype=int)])
    else:
        output = speech_tokenizer.run(
            None,
            {
                "feats": feat,
                "feats_length": np.array([feat.shape[2]], dtype=int),
            },
        )
    speech_token = output[0]

    speech_token_len = np.array([speech_token.shape[1]], dtype=int)
    return speech_token, speech_token_len


def extract_spk_embedding(campplus, speech):
    import torch

    speech = torch.from_numpy(speech)
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat = feat.numpy()

    feat = np.expand_dims(feat, axis=0)
    if not args.onnx:
        output = campplus.predict([feat])
    else:
        output = campplus.run(
            None,
            {"input": feat},
        )
    embedding = output[0]

    return embedding


def extract_speech_feat(speech):
    import torch

    speech = torch.from_numpy(speech)
    speech_feat = mel_spectrogram(
        speech,
        n_fft=1920,
        num_mels=80,
        sampling_rate=24000,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=8000,
        center=False,
    )
    speech_feat = speech_feat.numpy()

    speech_feat = np.squeeze(speech_feat, axis=0).transpose(1, 0)
    speech_feat = np.expand_dims(speech_feat, axis=0)
    speech_feat_len = np.array([speech_feat.shape[1]], dtype=int)

    return speech_feat, speech_feat_len


def inference(models, tts_text, prompt_text, prompt_speech_16k, speed=1.0):
    sample_rate = 24000

    logger.info("synthesis text {}".format(tts_text))
    start_time = time.time()

    tokenizer = models["tokenizer"]
    tokens = tokenizer([tts_text], return_tensors="np")
    tts_text_token = tokens["input_ids"]
    tts_text_token_len = np.array([tts_text_token.shape[1]], dtype=int)

    tokens = tokenizer([prompt_text], return_tensors="np")
    prompt_text_token = tokens["input_ids"]
    prompt_text_token_len = np.array([prompt_text_token.shape[1]], dtype=int)

    resample_rate = 24000
    prompt_speech_resample = librosa.resample(
        prompt_speech_16k, orig_sr=16000, target_sr=resample_rate
    )

    speech_feat, speech_feat_len = extract_speech_feat(prompt_speech_resample)

    speech_tokenizer = models["speech_tokenizer"]
    speech_token, speech_token_len = extract_speech_token(
        speech_tokenizer, prompt_speech_16k
    )

    # cosyvoice2, force speech_feat % speech_token = 2
    token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
    speech_feat, speech_feat_len[:] = speech_feat[:, : 2 * token_len], 2 * token_len
    speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len

    campplus = models["campplus"]
    embedding = extract_spk_embedding(campplus, prompt_speech_16k)

    model_input = {
        "text": tts_text_token,
        "text_len": tts_text_token_len,
        "prompt_text": prompt_text_token,
        "prompt_text_len": prompt_text_token_len,
        "llm_prompt_speech_token": speech_token,
        "llm_prompt_speech_token_len": speech_token_len,
        "flow_prompt_speech_token": speech_token,
        "flow_prompt_speech_token_len": speech_token_len,
        "prompt_speech_feat": speech_feat,
        "prompt_speech_feat_len": speech_feat_len,
        "llm_embedding": embedding,
        "flow_embedding": embedding,
    }

    # for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
    #     speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
    #     logger.info(
    #         "yield speech len {}, rtf {}".format(
    #             speech_len, (time.time() - start_time) / speech_len
    #         )
    #     )
    #     yield model_output
    #     start_time = time.time()


def inference_zero_shot(models):
    # prompt = args.input if isinstance(args.input, str) else args.input[0]
    audio_path = "common_voice_ja_37088095.wav"
    tts_text = "それから食事のときには、彼等のうちから数人招きます。もっともそのときには、普通の人間も、二三人ずつ立派な人を招くことにします。"
    prompt_text = "建築的な統一にもたらされることによって科学的となるのである。"
    prompt_speech_16k = np.load("prompt_speech_16k.npy")

    logger.info("tts_text: %s" % tts_text)
    logger.info("prompt_text: %s" % prompt_text)

    logger.info("Start inference...")

    tokenizer = models["tokenizer"]
    prompt_text = text_normalize(tokenizer, prompt_text, split=False)
    for s in text_normalize(tokenizer, tts_text, split=True):
        if len(s) < 0.5 * len(prompt_text):
            logger.warning(
                "synthesis text {} too short than prompt text {}, this may lead to bad performance".format(
                    s, prompt_text
                )
            )

        tts_speech = inference(models, s, prompt_text, prompt_speech_16k)

    logger.info("Script finished successfully.")


def main():
    env_id = args.env_id

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
        speech_tokenizer = ailia.Net(
            "speech_tokenizer_v2.onnx.prototxt",
            "speech_tokenizer_v2.onnx",
            env_id=env_id,
        )
        campplus = ailia.Net(
            "campplus.onnx.prototxt",
            "campplus.onnx",
            env_id=env_id,
        )
        embed_tokens = ailia.Net(
            "CosyVoice2-0.5B_embed_tokens.onnx.prototxt",
            "CosyVoice2-0.5B_embed_tokens.onnx",
            env_id=env_id,
        )
        llm = ailia.Net(
            "CosyVoice2-0.5B_llm.onnx.prototxt",
            "CosyVoice2-0.5B_llm.onnx",
            env_id=env_id,
        )
        flow_enc = ailia.Net(
            "CosyVoice2-0.5B_flow_encoder.onnx.prototxt",
            "CosyVoice2-0.5B_flow_encoder.onnx",
            env_id=env_id,
        )
        flow_dec = ailia.Net(
            "CosyVoice2-0.5B_flow.decoder.estimator.fp32.onnx.prototxt",
            "CosyVoice2-0.5B_flow.decoder.estimator.fp32.onnx",
            env_id=env_id,
        )
        hift = ailia.Net(
            "CosyVoice2-0.5B_hift.onnx.prototxt",
            "CosyVoice2-0.5B_hift.onnx",
            env_id=env_id,
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    args.disable_ailia_tokenizer = True
    if args.disable_ailia_tokenizer:
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained("./tokenizer")

        special_tokens = {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            # fmt: off
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
            # fmt: on
        }
        tokenizer.add_special_tokens(special_tokens)
    else:
        raise NotImplementedError

    models = dict(
        tokenizer=tokenizer,
        speech_tokenizer=speech_tokenizer,
        campplus=campplus,
        embed_tokens=embed_tokens,
        llm=llm,
        flow_enc=flow_enc,
        flow_dec=flow_dec,
        hift=hift,
    )

    # generate
    inference_zero_shot(models)


if __name__ == "__main__":
    main()

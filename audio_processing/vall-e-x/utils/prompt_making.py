import os
import logging
import langid
langid.set_languages(['en', 'zh', 'ja'])

import numpy as np
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from utils.g2p import PhonemeBpeTokenizer

from macros import *

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")

def make_prompt(name, audio_prompt_path, transcript=None, models=None, ort=False):
    codec = AudioTokenizer(models["encodec.onnx"], ort=ort)
    text_pr, lang_pr = make_transcript(transcript)

    # tokenize audio
    encoded_frames = tokenize_audio(codec, audio_prompt_path)
    audio_tokens = np.transpose(encoded_frames[0][0], axes = (0, 2, 1))

    # tokenize text
    phone, langs, phonemes = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens = [phone]

    message = f"Detected language: {lang_pr}\n Detected text {text_pr}\n"

    # save as npz file
    save_path = os.path.join("./customs/", f"{name}.npz")
    np.savez(save_path, audio_tokens=audio_tokens, text_tokens=text_tokens, lang_code=lang2code[lang_pr])
    logging.info(f"Successful. Prompt saved to {save_path}")


def make_transcript(transcript):
    text = transcript
    lang, _ = langid.classify(text)
    lang_token = lang2token[lang]
    text = lang_token + text + lang_token
    return text, lang
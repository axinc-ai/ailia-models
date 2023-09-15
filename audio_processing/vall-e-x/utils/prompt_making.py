import os
import torch
import torchaudio
import logging
import langid
langid.set_languages(['en', 'zh', 'ja'])

import numpy as np
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from utils.g2p import PhonemeBpeTokenizer

from macros import *

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

codec = AudioTokenizer()

def make_prompt(name, audio_prompt_path, transcript=None, models=None):
    #global model, text_collater, text_tokenizer, codec
    #wav_pr, sr = torchaudio.load(audio_prompt_path)
    # check length
    #if wav_pr.size(-1) / sr > 15:
    #    raise ValueError(f"Prompt too long, expect length below 15 seconds, got {wav_pr / sr} seconds.")
    #if wav_pr.size(0) == 2:
    #    wav_pr = wav_pr.mean(0, keepdim=True)
    text_pr, lang_pr = make_transcript(transcript)

    # tokenize audio
    encoded_frames = tokenize_audio(codec, audio_prompt_path)#(wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, langs = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

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
    torch.cuda.empty_cache()
    return text, lang
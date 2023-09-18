# coding: utf-8
import os
import logging
import langid
langid.set_languages(['en', 'zh', 'ja'])

import ailia
import ailia.audio

import time

import numpy as np
from utils.g2p import PhonemeBpeTokenizer

from macros import *

model = None

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")

from models.vallex import VALLE


def vocos_istft(x, y): # for onnx
    n_fft = 1280
    hop_length = 320
    win_length = 1280

    ailia_audio = True

    if not ailia_audio:
        import torch
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        S = (x + 1j * y)
        window = torch.hann_window(win_length)
        audio = torch.istft(S, n_fft, hop_length, win_length, window, center=True)
        audio = audio.squeeze().cpu().numpy()
    else:
        S = (x + 1j * y)
        audio = ailia.audio.inverse_spectrogram(S, hop_n=hop_length, win_n=win_length, win_type="hann", norm_type="torch")
        audio = np.real(audio).squeeze()

    return audio


def generate_audio(text, prompt=None, language='auto', accent='no-accent', benchmark = False, models = None, ort = False, logger = None, top_k = -100):
    global model, vocos, text_tokenizer, text_collater
    text = text.replace("\n", "").strip(" ")
    # detect language
    if language == "auto":
        language = langid.classify(text)[0]
    lang_token = lang2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # load prompt
    if prompt is not None:
        prompt_path = prompt
        if not os.path.exists(prompt_path):
            prompt_path = "./presets/" + prompt + ".npz"
        if not os.path.exists(prompt_path):
            prompt_path = "./customs/" + prompt + ".npz"
        if not os.path.exists(prompt_path):
            raise ValueError(f"Cannot find prompt {prompt}")
        prompt_data = np.load(prompt_path)
        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]

        # numpy to tensor
        audio_prompts = audio_prompts
        text_prompts = text_prompts
    else:
        audio_prompts = np.zeros((1, 0, NUM_QUANTIZERS))
        text_prompts = np.zeros((1, 0))
        lang_pr = lang if lang != 'mix' else 'en'

    enroll_x_lens = text_prompts.shape[-1]
    logger.info(f"synthesize text: {text}")
    phone_tokens, langs, phonemes = text_tokenizer.tokenize(text=f"_{text}".strip())
    logger.info(f"synthesize phonemes: {phonemes}")
    text_tokens = np.array([phone_tokens])
    text_tokens_lens = np.array([len(phone_tokens)])

    text_tokens = np.concatenate([text_prompts, text_tokens], axis=-1)
    text_tokens_lens += enroll_x_lens
    # accent control
    lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
    model = VALLE(models)
    encoded_frames = model.inference(
        text_tokens,
        text_tokens_lens,
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang,
        top_k=top_k,
        benchmark=benchmark,
        ort=ort,
        logger=logger
    )

    # Decode with Vocos
    frames = encoded_frames.transpose((2,0,1))

    #print("Impot vocos from onnx")
    vnet = models["vocos.onnx"]
    if benchmark:
        start = int(round(time.time() * 1000))
    x, y = vnet.run([frames])
    if benchmark:
        end = int(round(time.time() * 1000))
        print(f'ailia processing time {end - start} ms')
    samples = vocos_istft(x, y)

    return samples


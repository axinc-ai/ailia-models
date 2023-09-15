# coding: utf-8
import os
import torch
import logging
import langid
langid.set_languages(['en', 'zh', 'ja'])

import ailia
import time

import pathlib
import platform
if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
from data.collation import get_text_token_collater
from utils.g2p import PhonemeBpeTokenizer

from macros import *

model = None

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

from models.vallex import VALLE


def vocos_istft(x, y): # for onnx
    S = (x + 1j * y)
    n_fft = 1280
    hop_length = 320
    win_length = 1280
    window = torch.hann_window(win_length)
    print("istft settings", n_fft, hop_length, win_length, window)
    audio = torch.istft(S, n_fft, hop_length, win_length, window, center=True)
    return audio


@torch.no_grad()
def generate_audio(text, prompt=None, language='auto', accent='no-accent', benchmark = False, models = None, ort = False):
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
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    else:
        audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32)
        text_prompts = torch.zeros([1, 0]).type(torch.int32)
        lang_pr = lang if lang != 'mix' else 'en'

    enroll_x_lens = text_prompts.shape[-1]
    logging.info(f"synthesize text: {text}")
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
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
        benchmark=benchmark,
        ort=ort
    )

    # Decode with Vocos
    frames = encoded_frames.permute(2,0,1)

    #print("Impot vocos from onnx")
    vnet = ailia.Net(weight="onnx/vocos.onnx", env_id = 1, memory_mode = 11)
    if benchmark:
        start = int(round(time.time() * 1000))
    x, y = vnet.run([frames.numpy()])
    end = int(round(time.time() * 1000))
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if benchmark:
        print(f'ailia processing time {end - start} ms')
    samples = vocos_istft(x, y)

    return samples.squeeze().cpu().numpy()


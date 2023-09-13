# coding: utf-8
import os
import torch
from vocos import Vocos
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

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from data.collation import get_text_token_collater
from utils.g2p import PhonemeBpeTokenizer

from macros import *

model = None

codec = None

vocos = None

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()


from modules.embedding import TokenEmbedding, SinePositionalEmbedding
from typing import Dict, Iterator, List, Tuple, Union

class VALLE():
    def __init__(
        self,
    ):
        self.language_ID = {
            'en': 0,
            'zh': 1,
            'ja': 2,
        }
        nar_scale_factor = 1.0
        nar_d_model = int(N_DIM * nar_scale_factor)
        self.ar_text_embedding = TokenEmbedding(N_DIM, NUM_TEXT_TOKENS)  # W_x
        self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)
        self.ar_language_embedding = TokenEmbedding(N_DIM, len(self.language_ID))
        self.nar_language_embedding = TokenEmbedding(N_DIM, len(self.language_ID))
        self.nar_text_prenet = nn.Identity()
        self.nar_audio_prenet = nn.Identity()
        self.ar_text_prenet = nn.Identity()
        self.ar_audio_prenet = nn.Identity()
        self.ar_text_position = SinePositionalEmbedding(
            N_DIM,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            N_DIM,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.nar_text_position = SinePositionalEmbedding(
            nar_d_model,
            dropout=0.0,
            scale=False,
            alpha=False,
        )
        self.nar_audio_position = SinePositionalEmbedding(
            nar_d_model,
            dropout=0.1,
            scale=False,
            alpha=False,
        )
        self.ar_audio_prepend_bos = True
        self.num_quantizers = NUM_QUANTIZERS
        self.prefix_mode = PREFIX_MODE
        assert self.num_quantizers >= 1
        self.nar_stage_embeddings = nn.ModuleList(
            [
                TokenEmbedding(nar_d_model, 1)
                for i in range(self.num_quantizers - 1)
            ]
        )
        if self.num_quantizers > 1:
            self.nar_audio_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)]
                + [
                    TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS)
                    for i in range(self.num_quantizers - 1)
                ]
            )  # W_a

        self.ar_text_embedding.load_onnx("ar_text_embedding.onnx")
        self.nar_text_embedding.load_onnx("nar_text_embedding.onnx")
        for i in range(len(self.nar_audio_embeddings)):
            self.nar_audio_embeddings[i].load_onnx("nar_audio_embeddings_"+str(i)+".onnx")
        self.ar_language_embedding.load_onnx("ar_language_embedding.onnx")
        self.nar_language_embedding.load_onnx("nar_language_embedding.onnx")

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
        prompt_language: str = None,
        text_language: str = None,
        benchmark = True
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        # Add language embedding
        prompt_language_id = torch.LongTensor(np.array([self.language_ID[prompt_language]])).to(x.device)
        if isinstance(text_language, str):
            text_language_id = torch.LongTensor(np.array([self.language_ID[text_language]])).to(x.device)
        elif isinstance(text_language, List):
            text_language_id = torch.LongTensor(np.array([self.language_ID[tl] for tl in text_language])).to(x.device)
        x[:, :enroll_x_lens, :] += self.ar_language_embedding(prompt_language_id)
        x[:, enroll_x_lens:, :] += self.ar_language_embedding(text_language_id)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()
        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        max_len = 1024 # TBD
        #kv_cache = torch.zeros((12, 2, 1, 16, max_len, 64))
        kv_cache_numpy = np.zeros((12, 2, 1, 16, max_len, 64))
        offset = 0
        # torch.Size([1, 16, n, 64])が12レイヤー * 2ノード分ある

        use_kv_caching = True
        while True:
            if offset == 0:
                print("Impot audio_embedding from onnx")
                anet = ailia.Net(weight="audio_embedding.onnx", env_id = 1, memory_mode = 11)
            start = int(round(time.time() * 1000))
            y_pos = anet.run([y.numpy()])[0]
            end = int(round(time.time() * 1000))
            y_pos = torch.from_numpy(y_pos)
            if benchmark:
                print(f'ailia processing time {end - start} ms')
            print("audio_embedding", y_pos.shape)

            xy_pos = torch.concat([x, y_pos], dim=1)

            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0
            ).to(y.device)


            if use_kv_caching and offset>=1:#kv_cache is not None:
                xy_pos = xy_pos[:, [-1]] # 前回のトークンは1つ
            else:
                pass # initial prompt

            if offset == 0:
                print("Impot ar_decoder from onnx")
                net = ailia.Net(weight="ar_decoder.onnx", env_id = 1, memory_mode = 11)
            offset_tensor = np.zeros((1))
            offset_tensor[0] = offset
            start = int(round(time.time() * 1000))
            logits, kv_cache_numpy = net.run([xy_pos.numpy(), xy_attn_mask.numpy(), kv_cache_numpy, offset_tensor])
            end = int(round(time.time() * 1000))
            logits = torch.from_numpy(logits)
            if benchmark:
                print(f'ailia processing time {end - start} ms offset {offset}')

            offset = offset + xy_pos.shape[-2]

            #logits = self.ar_predict_layer(xy_dec[:, -1]) # require export
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1, temperature=temperature
            )

            if (
                torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
                or samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )

                print(f"VALL-E EOS [{prompts.shape[1]} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)

        # Non-AR Decoders
        y_emb = self.nar_audio_embeddings[0](
            y[:, int(self.ar_audio_prepend_bos) :]
        )

        if self.prefix_mode in [2, 4]:  # Exclude enrolled_phonemes
            enrolled_len = enroll_x_lens.max().item()
            # SOS + Synthesis Text + EOS
            text = torch.concat(
                [
                    text[:, :1],
                    text[:, enrolled_len - 1 :],
                ],
                dim=1,
            )
            text_len = text_len - (enrolled_len - 2)
            assert text.shape[0] == 1

        x = self.nar_text_embedding(text)
        # Add language embedding
        prompt_language_id = torch.LongTensor(np.array([self.language_ID[prompt_language]])).to(x.device)
        if isinstance(text_language, str):
            text_language_id = torch.LongTensor(np.array([self.language_ID[text_language]])).to(x.device)
        elif isinstance(text_language, List):
            text_language_id = torch.LongTensor(np.array([self.language_ID[tl] for tl in text_language])).to(x.device)
        x[:, :enroll_x_lens, :] += self.nar_language_embedding(prompt_language_id)
        x[:, enroll_x_lens:, :] += self.nar_language_embedding(text_language_id)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

            for i in range(0, self.num_quantizers - 1):
                embedding_layer = self.nar_audio_embeddings[1+i]
                
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                print("Impot nar_decoder from onnx "+str(i))
                if i == 0:
                    nar_decoder = ailia.Net(weight="nar_decoder.onnx", env_id = 1, memory_mode = 11)
                offset_tensor = np.zeros((1))
                offset_tensor[0] = i
                print(xy_pos.shape, offset_tensor.shape)
                xy_dec = nar_decoder.run([xy_pos.numpy(), offset_tensor])[0]
                end = int(round(time.time() * 1000))
                xy_dec = torch.from_numpy(xy_dec)
                if benchmark:
                    print(f'ailia processing time {end - start} ms')

                print("Impot nar_predict_layers from onnx")
                if i == 0:
                    nar_predict = ailia.Net(weight="nar_predict_layers.onnx", env_id = 1, memory_mode = 11)
                logits = nar_predict.run([xy_dec[:, text_len + prefix_len :].numpy(), offset_tensor])[0]
                end = int(round(time.time() * 1000))
                logits = torch.from_numpy(logits)
                if benchmark:
                    print(f'ailia processing time {end - start} ms')
                
                print(logits.shape)

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token

def export_vocos_istft(x, y): # for onnx
    S = (x + 1j * y)
    n_fft = 1280
    hop_length = 320
    win_length = 1280
    window = torch.hann_window(win_length)
    print("istft settings", n_fft, hop_length, win_length, window)
    audio = torch.istft(S, n_fft, hop_length, win_length, window, center=True)
    return audio


@torch.no_grad()
def generate_audio(text, prompt=None, language='auto', accent='no-accent'):
    global model, codec, vocos, text_tokenizer, text_collater
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
    model = VALLE()
    encoded_frames = model.inference(
        text_tokens,
        text_tokens_lens,
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=-100,
        temperature=1,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang
    )

    # Decode with Vocos
    frames = encoded_frames.permute(2,0,1)

    print("Impot vocos from onnx")
    vnet = ailia.Net(weight="vocos.onnx", env_id = 1, memory_mode = 11)
    start = int(round(time.time() * 1000))
    x, y = vnet.run([frames.numpy()])
    end = int(round(time.time() * 1000))
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    benchmark = True
    if benchmark:
        print(f'ailia processing time {end - start} ms')
    samples = export_vocos_istft(x, y)

    return samples.squeeze().cpu().numpy()


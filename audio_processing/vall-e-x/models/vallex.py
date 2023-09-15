from modules.embedding import TokenEmbedding, TokenEmbeddingLayers, SinePositionalEmbedding
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import ailia
import time

from macros import *

class VALLE():
    def __init__(
        self,
        models
    ):
        self.language_ID = {
            'en': 0,
            'zh': 1,
            'ja': 2,
        }
        self.ar_audio_embedding = TokenEmbedding()
        self.ar_text_embedding = TokenEmbedding()
        self.nar_text_embedding = TokenEmbedding()
        self.ar_language_embedding = TokenEmbedding()
        self.nar_language_embedding = TokenEmbedding()
        self.nar_audio_embedding_layers = TokenEmbeddingLayers()
        self.nar_text_prenet = nn.Identity()
        self.nar_audio_prenet = nn.Identity()
        self.ar_text_prenet = nn.Identity()
        self.ar_audio_prenet = nn.Identity()
        self.ar_text_position = SinePositionalEmbedding(
            alpha_parameter=2.0744
        )
        self.ar_audio_position = SinePositionalEmbedding(
            alpha_parameter=2.3490
        )
        self.nar_text_position = SinePositionalEmbedding(
            alpha_parameter=1.0
        )
        self.nar_audio_position = SinePositionalEmbedding(
            alpha_parameter=1.0
        )
        self.ar_audio_prepend_bos = True
        self.num_quantizers = NUM_QUANTIZERS
        if self.num_quantizers > 1:
            self.nar_audio_embedding = TokenEmbedding()

        self.ar_audio_embedding.load_onnx(models["ar_audio_embedding.onnx"])
        self.ar_text_embedding.load_onnx(models["ar_text_embedding.onnx"])
        self.nar_text_embedding.load_onnx(models["nar_text_embedding.onnx"])
        self.nar_audio_embedding.load_onnx(models["nar_audio_embedding.onnx"])
        self.nar_audio_embedding_layers.load_onnx(models["nar_audio_embedding_layers.onnx"])
        self.ar_language_embedding.load_onnx(models["ar_language_embedding.onnx"])
        self.nar_language_embedding.load_onnx(models["nar_language_embedding.onnx"])

        self.ar_text_position.load_onnx(models["position_embedding.onnx"])
        self.ar_audio_position.load_onnx(models["position_embedding.onnx"])
        self.nar_text_position.load_onnx(models["position_embedding.onnx"])
        self.nar_audio_position.load_onnx(models["position_embedding.onnx"])

        self.models = models

    def audio_embedding(self, y):
        y_emb = torch.tensor(self.ar_audio_embedding.forward(y.numpy()))
        y_emb = self.ar_audio_prenet(y_emb)
        y_pos = torch.tensor(self.ar_audio_position.forward(y_emb.numpy()))
        return y_pos

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        prompt_language: str = None,
        text_language: str = None,
        benchmark = False,
        ort = False
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
        x = torch.tensor(self.ar_text_embedding.forward(text.numpy()))
        # Add language embedding
        print(x.device)
        prompt_language_id = torch.LongTensor(np.array([self.language_ID[prompt_language]])).to(x.device)
        if isinstance(text_language, str):
            text_language_id = torch.LongTensor(np.array([self.language_ID[text_language]])).to(x.device)
        elif isinstance(text_language, List):
            text_language_id = torch.LongTensor(np.array([self.language_ID[tl] for tl in text_language])).to(x.device)
        x[:, :enroll_x_lens, :] += torch.tensor(self.ar_language_embedding.forward(prompt_language_id.numpy()))
        x[:, enroll_x_lens:, :] += torch.tensor(self.ar_language_embedding.forward(text_language_id.numpy()))
        x = self.ar_text_prenet(x)
        x = torch.tensor(self.ar_text_position.forward(x.numpy()))

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
        kv_cache_numpy = np.zeros((12 * 2, 16, max_len, 64), dtype=np.float32)   # torch.Size([1, 16, n, 64])が12レイヤー * 2ノード分ある
        offset = 0
       
        use_kv_caching = True
        while True:
            y_pos = self.audio_embedding(y)
           
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
            
            if "ar_decoder.opt.onnx" in self.models:
                net = self.models["ar_decoder.opt.onnx"]
            else:
                net = self.models["ar_decoder.onnx"]
            offset_tensor = np.array(offset, dtype=np.int64) # constant type (shape = ())
            start = int(round(time.time() * 1000))
            if ort:
                logits, kv_cache_numpy = net.run(None, {"xy_pos":xy_pos.numpy(), "mask":xy_attn_mask.numpy(), "past_kv":kv_cache_numpy, "offset":offset_tensor})
            else:
                if offset == 0:
                    logits, kv_cache_numpy = net.run({"xy_pos":xy_pos.numpy(), "mask":xy_attn_mask.numpy(), "past_kv":kv_cache_numpy, "offset":offset_tensor})
                else:
                    logits = np.zeros((1, 1025), dtype=np.float32, order='C')
                    output = [logits]
                    net.copy_blob_data(net.find_blob_index_by_name("past_kv"), net.find_blob_index_by_name("kv_cache"), None)
                    net.run({"xy_pos":xy_pos.numpy(), "mask":xy_attn_mask.numpy(), "offset":offset_tensor}, output = output)
            end = int(round(time.time() * 1000))
            logits = torch.from_numpy(logits)
            if benchmark:
                print(f'ailia processing time {end - start} ms offset {offset}')

            offset = offset + xy_pos.shape[-2]

            #logits = self.ar_predict_layer(xy_dec[:, -1]) # require export
            samples = topk_sampling(
                logits
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
        y_emb = torch.tensor(self.nar_audio_embedding.forward(
            y[:, int(self.ar_audio_prepend_bos) :].numpy()
        ))

        x = torch.tensor(self.nar_text_embedding.forward(text.numpy()))
        # Add language embedding
        prompt_language_id = torch.LongTensor(np.array([self.language_ID[prompt_language]])).to(x.device)
        if isinstance(text_language, str):
            text_language_id = torch.LongTensor(np.array([self.language_ID[text_language]])).to(x.device)
        elif isinstance(text_language, List):
            text_language_id = torch.LongTensor(np.array([self.language_ID[tl] for tl in text_language])).to(x.device)
        x[:, :enroll_x_lens, :] += torch.tensor(self.nar_language_embedding.forward(prompt_language_id.numpy()))
        x[:, enroll_x_lens:, :] += torch.tensor(self.nar_language_embedding.forward(text_language_id.numpy()))
        x = self.nar_text_prenet(x)
        x = torch.tensor(self.nar_text_position.forward(x.numpy()))

        for j in range(1, self.num_quantizers):
            if prefix_len > 0:
                y_emb[:, :prefix_len] += torch.tensor(self.nar_audio_embedding_layers.forward(
                    prompts[..., j].numpy(), j - 1
                ))

        for i in range(0, self.num_quantizers - 1):
            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = torch.tensor(self.nar_audio_position.forward(y_pos.numpy()))
            xy_pos = torch.concat([x, y_pos], dim=1)

            #print("Impot nar_decoder from onnx "+str(i))
            if "nar_decoder.opt.onnx" in self.models:
                nar_decoder = self.models["nar_decoder.opt.onnx"]
            else:
                nar_decoder = self.models["nar_decoder.onnx"]
            offset_tensor = np.array(i, dtype=np.int64) # constant type (shape = ())
            #print(xy_pos.shape, offset_tensor.shape)
            if benchmark:
                start = int(round(time.time() * 1000))
            xy_dec = nar_decoder.run([xy_pos.numpy(), offset_tensor])[0]
            if benchmark:
                end = int(round(time.time() * 1000))
            xy_dec = torch.from_numpy(xy_dec)
            if benchmark:
                print(f'ailia processing time {end - start} ms')

            #print("Impot nar_predict_layers from onnx")
            if i == 0:
                nar_predict = self.models["nar_predict_layers.onnx"]
            if benchmark:
                start = int(round(time.time() * 1000))
            logits = nar_predict.run([xy_dec[:, text_len + prefix_len :].numpy(), offset_tensor])[0]
            if benchmark:
                end = int(round(time.time() * 1000))
            logits = torch.from_numpy(logits)
            if benchmark:
                print(f'ailia processing time {end - start} ms')
            
            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)

            if i < self.num_quantizers - 2:
                y_emb[:, prefix_len:] += torch.tensor(self.nar_audio_embedding_layers.forward(samples.numpy(), i))

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)


def topk_sampling(logits):
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token
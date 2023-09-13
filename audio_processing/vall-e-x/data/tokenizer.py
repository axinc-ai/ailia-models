#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import ailia
import numpy as np
import torch
import torchaudio
from encodec.utils import convert_audio


class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self
    ) -> None:
        self.sample_rate = 24000
        self.channels = 1
        self.vnet = ailia.Net(weight="onnx/encodec.onnx", env_id = 1, memory_mode = 11)
        

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        #print("Impot encodec from onnx")
        encoded_frames = self.vnet.run([wav.numpy()])[0]
        encoded_frames = torch.from_numpy(encoded_frames)
        encoded_frames = [[encoded_frames]]

        return encoded_frames



def tokenize_audio(tokenizer: AudioTokenizer, audio):
    # Load and pre-process the audio waveform
    if isinstance(audio, str):
        wav, sr = torchaudio.load(audio)
    else:
        wav, sr = audio
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)

    return encoded_frames


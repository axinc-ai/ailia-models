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
import librosa

class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        net
    ) -> None:
        self.sample_rate = 24000
        self.channels = 1
        self.net = net
        

    def encode(self, wav):
        encoded_frames = self.net.run([wav])[0]
        encoded_frames = [[encoded_frames]]
        return encoded_frames



def tokenize_audio(tokenizer: AudioTokenizer, audio):
    wav, sr = librosa.load(audio, sr=tokenizer.sample_rate, mono=True)
    wav = np.array(wav)
    wav = np.expand_dims(wav, axis = 0)
    wav = np.expand_dims(wav, axis = 0)

    if wav.shape[-1] / sr > 15:
        raise ValueError(f"Prompt too long, expect length below 15 seconds, got {wav.shape[-1] / sr} seconds.")

    # Extract discrete codes from EnCodec
    encoded_frames = tokenizer.encode(wav)
    return encoded_frames


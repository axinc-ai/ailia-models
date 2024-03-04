# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, reduce
from torchaudio.transforms import MFCC

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)


class SimpleEmbeddingModel(Model):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.mfcc = MFCC(
            sample_rate=self.hparams.sample_rate,
            n_mfcc=40,
            dct_type=2,
            norm="ortho",
            log_mels=False,
        )

        self.lstm = nn.LSTM(
            self.mfcc.n_mfcc * self.hparams.num_channels,
            32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames

        Source
        ------
        https://pytorch.org/docs/stable/generated/torch.stft.html#torch.stft

        """

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft
        center = self.mfcc.MelSpectrogram.spectrogram.center

        return conv1d_num_frames(
            num_samples=num_samples,
            kernel_size=n_fft,
            stride=hop_length,
            padding=n_fft // 2 if center else 0,
            dilation=1,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft

        return conv1d_receptive_field_size(
            num_frames, kernel_size=n_fft, stride=hop_length, dilation=1
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        hop_length = self.mfcc.MelSpectrogram.spectrogram.hop_length
        n_fft = self.mfcc.MelSpectrogram.spectrogram.n_fft
        center = self.mfcc.MelSpectrogram.spectrogram.center

        return conv1d_receptive_field_center(
            frame=frame,
            kernel_size=n_fft,
            stride=hop_length,
            padding=n_fft // 2 if center else 0,
            dilation=1,
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return 64

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : (batch, time, channel)

        Returns
        -------
        embedding : (batch, dimension)
        """

        mfcc = self.mfcc(waveforms)
        output, hidden = self.lstm(rearrange(mfcc, "b c f t -> b t (c f)"))
        # mean temporal pooling
        return reduce(output, "b t f -> b f", "mean")

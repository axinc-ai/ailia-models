# MIT License
#
# Copyright (c) 2023 CNRS
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


from functools import lru_cache, partial
from typing import Optional

import torch
import torchaudio.compliance.kaldi as kaldi

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)

from .resnet import ResNet34, ResNet152, ResNet221, ResNet293


class BaseWeSpeakerResNet(Model):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.save_hyperparameters(
            "sample_rate",
            "num_channels",
            "num_mel_bins",
            "frame_length",
            "frame_shift",
            "dither",
            "window_type",
            "use_energy",
        )

        self._fbank = partial(
            kaldi.fbank,
            num_mel_bins=self.hparams.num_mel_bins,
            frame_length=self.hparams.frame_length,
            frame_shift=self.hparams.frame_shift,
            dither=self.hparams.dither,
            sample_frequency=self.hparams.sample_rate,
            window_type=self.hparams.window_type,
            use_energy=self.hparams.use_energy,
        )

    def compute_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract fbank features

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)

        Returns
        -------
        fbank : (batch_size, num_frames, num_mel_bins)

        Source: https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/bin/infer_onnx.py#L30C1-L50
        """

        waveforms = waveforms * (1 << 15)

        # fall back to CPU for FFT computation when using MPS
        # until FFT is fixed in MPS
        device = waveforms.device
        fft_device = torch.device("cpu") if device.type == "mps" else device

        features = torch.vmap(self._fbank)(waveforms.to(fft_device)).to(device)

        return features - torch.mean(features, dim=1, keepdim=True)

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        return self.resnet.embed_dim

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """
        window_size = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
        step_size = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)

        num_frames = conv1d_num_frames(
            num_samples=num_samples,
            kernel_size=window_size,
            stride=step_size,
            padding=0,
            dilation=1,
        )
        return self.resnet.num_frames(num_frames)

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
        receptive_field_size = num_frames
        receptive_field_size = self.resnet.receptive_field_size(receptive_field_size)

        window_size = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
        step_size = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)

        return conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=window_size,
            stride=step_size,
            dilation=1,
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
        receptive_field_center = frame
        receptive_field_center = self.resnet.receptive_field_center(
            frame=receptive_field_center
        )

        window_size = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
        step_size = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)
        return conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=window_size,
            stride=step_size,
            padding=0,
            dilation=1,
        )

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of waveforms with shape (batch, channel, sample)
        weights : torch.Tensor, optional
            Batch of weights with shape (batch, frame).
        """

        fbank = self.compute_fbank(waveforms)
        return self.resnet(fbank, weights=weights)[1]


class WeSpeakerResNet34(BaseWeSpeakerResNet):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            task=task,
        )
        self.resnet = ResNet34(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )


class WeSpeakerResNet152(BaseWeSpeakerResNet):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            task=task,
        )
        self.resnet = ResNet152(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )


class WeSpeakerResNet221(BaseWeSpeakerResNet):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            task=task,
        )
        self.resnet = ResNet221(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )


class WeSpeakerResNet293(BaseWeSpeakerResNet):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        window_type: str = "hamming",
        use_energy: bool = False,
        task: Optional[Task] = None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_channels=num_channels,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            dither=dither,
            window_type=window_type,
            use_energy=use_energy,
            task=task,
        )
        self.resnet = ResNet293(
            num_mel_bins, 256, pooling_func="TSTP", two_emb_layer=False
        )


__all__ = [
    "WeSpeakerResNet34",
    "WeSpeakerResNet152",
    "WeSpeakerResNet221",
    "WeSpeakerResNet293",
]

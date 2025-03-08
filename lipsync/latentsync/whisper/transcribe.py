from typing import Optional, Union

import numpy as np
import torch
import tqdm

from .audio import N_FRAMES, pad_or_trim, log_mel_spectrogram


def transcribe(
    model,
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16

    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    mel = log_mel_spectrogram(audio)

    all_segments = []

    def add_segment(*, start: float, end: float, encoder_embeddings):
        all_segments.append(
            {
                "start": start,
                "end": end,
                "encoder_embeddings": encoder_embeddings,
            }
        )

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    seek = 0
    sample_skip = 3000
    with tqdm.tqdm(total=num_frames, unit="frames", disable=verbose is not False):
        while seek < num_frames:
            # seek是开始的帧数
            end_seek = min(seek + sample_skip, num_frames)
            segment = (
                pad_or_trim(mel[:, seek : seek + sample_skip], N_FRAMES)
                .to(model.device)
                .to(dtype)
            )

            single = segment.ndim == 2
            if single:
                segment = segment.unsqueeze(0)
            if dtype == torch.float16:
                segment = segment.half()

            audio_features, embeddings = model.encoder(segment, include_embeddings=True)
            embeddings = embeddings.detach().cpu().numpy()

            encoder_embeddings = embeddings
            add_segment(
                start=seek,
                end=end_seek,
                encoder_embeddings=encoder_embeddings,
            )
            seek += sample_skip

    return dict(segments=all_segments)

import torch
import torchaudio


def load_audio(audio_path: str, sample_rate: int) -> torch.tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) != 1:
        raise ValueError("Only mono audio is supported")

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    # Since our model requires a batch dimension,
    # treat the unneeded audio channel dimension (1) as the batch dimension
    return wav

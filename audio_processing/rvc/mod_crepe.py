import functools

import numpy as np

from functional import im2col

CENTS_PER_BIN = 20  # cents
MAX_FMAX = 2006.  # hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples
UNVOICED = np.nan


def viterbi(logits):
    """Sample observations using viterbi decoding"""
    # Create viterbi transition matrix
    if not hasattr(viterbi, 'transition'):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        viterbi.transition = transition

    # Normalize logits
    with torch.no_grad():
        probs = torch.nn.functional.softmax(logits, dim=1)

    # Convert to numpy
    sequences = probs.cpu().numpy()

    # Perform viterbi decoding
    bins = np.array([
        librosa.sequence.viterbi(sequence, viterbi.transition).astype(np.int64)
        for sequence in sequences])

    # Convert to pytorch
    bins = torch.tensor(bins, device=probs.device)

    # Convert to frequency in Hz
    return bins, torchcrepe.convert.bins_to_frequency(bins)


def predict(
        audio,
        sample_rate,
        hop_length=None,
        fmin=50.,
        fmax=MAX_FMAX,
        decoder=viterbi,
        return_periodicity=False,
        batch_size=None,
        pad=True):
    """Performs pitch estimation

    Arguments
        audio (torch.tensor [shape=(1, time)])
            The audio signal
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_harmonicity (bool) [DEPRECATED]
            Whether to also return the network confidence
        return_periodicity (bool)
            Whether to also return the network confidence
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        pitch (torch.tensor [shape=(1, 1 + int(time // hop_length))])
        (Optional) periodicity (torch.tensor
                                [shape=(1, 1 + int(time // hop_length))])
    """

    results = []

    # Preprocess audio
    generator = preprocess(
        audio, sample_rate, hop_length, batch_size, pad)

    for frames in generator:
        # Infer independent probabilities for each pitch bin
        probabilities = infer(frames, model)

        # shape=(batch, 360, time / hop_length)
        probabilities = probabilities.reshape(
            audio.size(0), -1, PITCH_BINS).transpose(1, 2)

        # Convert probabilities to F0 and periodicity
        result = postprocess(
            probabilities, fmin, fmax,
            decoder, return_harmonicity, return_periodicity)

        # Place on same device as audio to allow very long inputs
        if isinstance(result, tuple):
            result = (result[0].to(audio.device),
                      result[1].to(audio.device))
        else:
            result = result.to(audio.device)

        results.append(result)

    # Split pitch and periodicity
    if return_periodicity:
        pitch, periodicity = zip(*results)
        return torch.cat(pitch, 1), torch.cat(periodicity, 1)

    # Concatenate
    return torch.cat(results, 1)


def postprocess(
        probabilities,
        fmin=0.,
        fmax=MAX_FMAX,
        decoder=viterbi,
        return_periodicity=False):
    """Convert model output to F0 and periodicity

    Arguments
        probabilities (torch.tensor [shape=(1, 360, time / hop_length)])
            The probabilities for each pitch bin inferred by the network
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        viterbi (bool)
            Whether to use viterbi decoding
        return_periodicity (bool)
            Whether to also return the network confidence

    Returns
        pitch (torch.tensor [shape=(1, 1 + int(time // hop_length))])
        periodicity (torch.tensor [shape=(1, 1 + int(time // hop_length))])
    """
    # Sampling is non-differentiable, so remove from graph
    probabilities = probabilities.detach()

    # Convert frequency range to pitch bin range
    minidx = torchcrepe.convert.frequency_to_bins(torch.tensor(fmin))
    maxidx = torchcrepe.convert.frequency_to_bins(
        torch.tensor(fmax), torch.ceil)

    # Remove frequencies outside of allowable range
    probabilities[:, :minidx] = -float('inf')
    probabilities[:, maxidx:] = -float('inf')

    # Perform argmax or viterbi sampling
    bins, pitch = decoder(probabilities)

    if not return_periodicity:
        return pitch

    # Compute periodicity from probabilities and decoded pitch bins
    return pitch, periodicity(probabilities, bins)


def preprocess(
        audio,
        sample_rate,
        hop_length=None,
        batch_size=None,
        pad=True):
    """Convert audio to model input

    Arguments
        audio (np.ndarray [shape=(1, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        frames (np.ndarray [shape=(1 + int(time // hop_length), 1024)])
    """
    # Default hop length of 10 ms
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Resample
    if sample_rate != SAMPLE_RATE:
        # We have to use resampy if we want numbers to match Crepe
        import resampy

        audio = audio[0]
        audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)
        audio = audio[None]
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    # Get total number of frames

    # Maybe pad
    if pad:
        total_frames = 1 + int(audio.shape[1] // hop_length)
        audio = np.pad(
            audio,
            ((0, 0), (WINDOW_SIZE // 2, WINDOW_SIZE // 2)))
    else:
        total_frames = 1 + int((audio.shape[1] - WINDOW_SIZE) // hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):
        # Batch indices
        start = max(0, i * hop_length)
        end = min(
            audio.shape[1],
            (i + batch_size - 1) * hop_length + WINDOW_SIZE)

        kernel_size = (1, WINDOW_SIZE)
        stride = (1, hop_length)
        unfold = functools.partial(im2col, filters=kernel_size, stride=stride)

        # Chunk
        frames, *_ = unfold(audio[:, None, None, start:end])

        # shape=(1 + int(time / hop_length, 1024)
        frames = frames[None].transpose(0, 2, 1).reshape(-1, WINDOW_SIZE)

        # Mean-center
        frames -= np.mean(frames, axis=1, keepdims=True)

        # Scale
        # Note: during silent frames, this produces very large values. But
        # this seems to be what the network expects.
        std = np.std(frames, axis=1, keepdims=True)
        frames /= np.where(std > 1e-10, std, 1e-10)

        yield frames

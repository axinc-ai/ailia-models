import os
import sys
import time
import io

import numpy as np
import cv2
from scipy.io import wavfile
import pydub
import torch
import torchaudio
import transformers

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, urlretrieve, progress_print  # noqa
# logger
from logging import getLogger  # noqa

import df

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_UNET_PATH = 'unet.onnx'
WEIGHT_PB_UNET_PATH = 'weights.pb'
MODEL_UNET_PATH = 'unet.onnx.prototxt'
WEIGHT_SAFETY_CHECKER_PATH = 'safety_checker.onnx'
MODEL_SAFETY_CHECKER_PATH = 'safety_checker.onnx.prototxt'
WEIGHT_TEXT_ENCODER_PATH = 'text_encoder.onnx'
MODEL_TEXT_ENCODER_PATH = 'text_encoder.onnx.prototxt'
WEIGHT_VAE_ENCODER_PATH = 'vae_encoder.onnx'
MODEL_VAE_ENCODER_PATH = 'vae_encoder.onnx.prototxt'
WEIGHT_VAE_DECODER_PATH = 'vae_decoder.onnx'
MODEL_VAE_DECODER_PATH = 'vae_decoder.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/anything_v3/'

SAVE_IMAGE_PATH = 'output.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Riffusion', None, SAVE_IMAGE_PATH
)
parser.add_argument(
    "-i", "--input", metavar="TEXT", type=str,
    default="pikachu",
    help="the prompt to render"
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser, check_input_type=False)


# ======================
# Secondaty Functions
# ======================

def audio_from_waveform(
        samples: np.ndarray, sample_rate: int, normalize: bool = False) -> pydub.AudioSegment:
    """
    Convert a numpy array of samples of a waveform to an audio segment.

    Args:
        samples: (channels, samples) array
    """
    # Normalize volume to fit in int16
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))

    # Transpose and convert to int16
    samples = samples.transpose(1, 0)
    samples = samples.astype(np.int16)

    # Write to the bytes of a WAV file
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)

    # Read into pydub
    return pydub.AudioSegment.from_wav(wav_bytes)


def audio_filters(segment: pydub.AudioSegment) -> pydub.AudioSegment:
    """
    Apply post-processing filters to the audio segment to compress it and
    keep at a -10 dBFS level.
    """
    # TODO(hayk): Come up with a principled strategy for these filters and experiment end-to-end.
    # TODO(hayk): Is this going to make audio unbalanced between sequential clips?

    desired_db = -12
    segment = segment.apply_gain(desired_db - segment.dBFS)

    segment = pydub.effects.normalize(
        segment,
        headroom=0.1,
    )

    return segment


# ======================
# Main functions
# ======================

class SpectrogramConverter:
    """
    Convert between audio segments and spectrogram tensors using torchaudio.

    In this class a "spectrogram" is defined as a (batch, time, frequency) tensor with float values
    that represent the amplitude of the frequency at that time bucket (in the frequency domain).
    Frequencies are given in the perceptul Mel scale defined by the params. A more specific term
    used in some functions is "mel amplitudes".

    The spectrogram computed from `spectrogram_from_audio` is complex valued, but it only
    returns the amplitude, because the phase is chaotic and hard to learn. The function
    `audio_from_spectrogram` is an approximate inverse of `spectrogram_from_audio`, which
    approximates the phase information using the Griffin-Lim algorithm.

    Each channel in the audio is treated independently, and the spectrogram has a batch dimension
    equal to the number of channels in the input audio segment.

    Both the Griffin Lim algorithm and the Mel scaling process are lossy.

    For more information, see https://pytorch.org/audio/stable/transforms.html
    """

    def __init__(self, params):
        self.p = params

        # # https://pytorch.org/audio/stable/generated/torchaudio.transforms.Spectrogram.html
        # self.spectrogram_func = torchaudio.transforms.Spectrogram(
        #     n_fft=params["n_fft"],
        #     hop_length=params["hop_length"],
        #     win_length=params["win_length"],
        #     pad=0,
        #     window_fn=torch.hann_window,
        #     power=None,
        #     normalized=False,
        #     wkwargs=None,
        #     center=True,
        #     pad_mode="reflect",
        #     onesided=True,
        # )

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html
        self.inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params["n_fft"],
            n_iter=params["num_griffin_lim_iters"],
            win_length=params["win_length"],
            hop_length=params["hop_length"],
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        )

        # # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelScale.html
        # self.mel_scaler = torchaudio.transforms.MelScale(
        #     n_mels=params["num_frequencies"],
        #     sample_rate=params["sample_rate"],
        #     f_min=params["min_frequency"],
        #     f_max=params["max_frequency"],
        #     n_stft=params["n_fft"] // 2 + 1,
        #     norm=params["mel_scale_norm"],
        #     mel_scale=params["mel_scale_type"],
        # )

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html
        self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params["n_fft"] // 2 + 1,
            n_mels=params["num_frequencies"],
            sample_rate=params["sample_rate"],
            f_min=params["min_frequency"],
            f_max=params["max_frequency"],
            max_iter=params["max_mel_iters"],
            tolerance_loss=1e-5,
            tolerance_change=1e-8,
            sgdargs=None,
            norm=params["mel_scale_norm"],
            mel_scale=params["mel_scale_type"],
        )

    def audio_from_spectrogram(
            self,
            spectrogram: np.ndarray,
            apply_filters: bool = True,
    ) -> pydub.AudioSegment:
        """
        Reconstruct an audio segment from a spectrogram.

        Args:
            spectrogram: (batch, frequency, time)
            apply_filters: Post-process with normalization and compression

        Returns:
            audio: Audio segment with channels equal to the batch dimension
        """
        # Move to device
        amplitudes_mel = torch.from_numpy(spectrogram)

        # Reconstruct the waveform
        waveform = self.waveform_from_mel_amplitudes(amplitudes_mel)

        # Convert to audio segment
        segment = audio_from_waveform(
            samples=waveform.cpu().numpy(),
            sample_rate=self.p["sample_rate"],
            # Normalize the waveform to the range [-1, 1]
            normalize=True,
        )

        # Optionally apply post-processing filters
        if apply_filters:
            segment = audio_filters(segment)

        return segment

    def waveform_from_mel_amplitudes(
            self,
            amplitudes_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-only function to approximately reconstruct a waveform from Mel-scale amplitudes.

        Args:
            amplitudes_mel: (batch, frequency, time)

        Returns:
            waveform: (batch, samples)
        """
        # Convert from mel scale to linear
        amplitudes_linear = self.inverse_mel_scaler(amplitudes_mel)

        # Run the approximate algorithm to compute the phase and recover the waveform
        waveform = self.inverse_spectrogram_func(amplitudes_linear)
        return waveform


def spectrogram_from_image(
        image: np.ndarray,
        power: float = 0.25,
        stereo: bool = False,
        max_value: float = 30e6) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    This is the inverse of image_from_spectrogram, except for discretization error from
    quantizing to uint8.

    Args:
        image: (frequency, time, channels)
        power: The power curve applied to the spectrogram
        stereo: Whether the spectrogram encodes stereo data
        max_value: The max value of the original spectrogram. In practice doesn't matter.

    Returns:
        spectrogram: (channels, frequency, time)
    """

    # Flip Y
    image = image[::-1, :, :]

    # Munge channels into a numpy array of (channels, frequency, time)
    data = image.transpose(2, 0, 1)
    if stereo:
        # Take the G and B channels as done in image_from_spectrogram
        data = data[[1, 2], :, :]
    else:
        data = data[0:1, :, :]

    # Convert to floats
    data = data.astype(np.float32)

    # Invert
    data = 255 - data

    # Rescale to 0-1
    data = data / 255

    # Reverse the power curve
    data = np.power(data, 1 / power)

    # Rescale to max value
    data = data * max_value

    return data


def audio_from_spectrogram_image(
        converter,
        image,
        apply_filters: bool = True,
        max_value: float = 30e6):
    """
    Reconstruct an audio segment from a spectrogram image.

    Args:
        image: Spectrogram image (in pillow format)
        apply_filters: Apply post-processing to improve the reconstructed audio
        max_value: Scaled max amplitude of the spectrogram. Shouldn't matter.
    """
    spectrogram = spectrogram_from_image(
        image,
        max_value=max_value,
        power=converter.p["power_for_image"],
        stereo=converter.p["stereo"],
    )

    segment = converter.audio_from_spectrogram(
        spectrogram,
        apply_filters=apply_filters,
    )

    return segment


def recognize_from_text(pipe):
    # prompt = args.input if isinstance(args.input, str) else args.input[0]
    prompt = "jazzy rapping from paris"
    negative_prompt = ""
    num_clips = 1
    num_inference_steps = 30
    guidance = 7.0
    width = 512
    seed = 42
    logger.info("prompt: %s" % prompt)

    logger.info('Start inference...')

    image = pipe.forward(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        negative_prompt=negative_prompt or None,
        width=width,
        height=512,
    )
    image = (image[0] * 255).astype(np.uint8)

    converter = SpectrogramConverter(params=dict(
        n_fft=17640,
        hop_length=441,
        win_length=4410,
        num_frequencies=512,
        sample_rate=44100,
        min_frequency=0,
        max_frequency=10000,
        max_mel_iters=200,
        mel_scale_norm=None,
        mel_scale_type="htk",
        num_griffin_lim_iters=32,
        power_for_image=0.25,
        stereo=False,
    ))
    segment = audio_from_spectrogram_image(converter, image)

    savepath = get_savepath(args.savepath, "", ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, image)

    savepath = get_savepath("output.wav", "", ext='.wav')
    logger.info(f'saved at : {savepath}')
    segment.export(savepath, format="wav")

    logger.info('Script finished successfully.')


def main():
    # check_and_download_models(WEIGHT_UNET_PATH, MODEL_UNET_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_SAFETY_CHECKER_PATH, MODEL_SAFETY_CHECKER_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, REMOTE_PATH)

    if not os.path.exists(WEIGHT_PB_UNET_PATH):
        logger.info('Downloading weights.pb...')
        urlretrieve(REMOTE_PATH, WEIGHT_PB_UNET_PATH, progress_print)
    logger.info('weights.pb is prepared!')

    env_id = args.env_id

    # initialize
    if not args.onnx:
        pass
    else:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

        net = onnxruntime.InferenceSession("unet.onnx", providers=providers)
        vae_decoder = onnxruntime.InferenceSession("vae_decoder.onnx", providers=providers)
        text_encoder = onnxruntime.InferenceSession("text_encoder.onnx", providers=providers)

    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        "./tokenizer"
    )
    scheduler = df.schedulers.DPMSolverMultistepScheduler.from_config({
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "trained_betas": None,
        "skip_prk_steps": True,
        "set_alpha_to_one": False,
        "prediction_type": "epsilon",
        "steps_offset": 1,
    })

    pipe = df.StableDiffusion(
        # vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=net,
        scheduler=scheduler,
        # safety_checker=safety_checker,
        # feature_extractor=feature_extractor,
        # requires_safety_checker=True
        use_onnx=args.onnx,
    )

    # generate
    recognize_from_text(pipe)


if __name__ == '__main__':
    main()

import os
import sys
import time

import numpy as np
import cv2
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

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.Spectrogram.html
        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            n_fft=params["n_fft"],
            hop_length=params["hop_length"],
            win_length=params["win_length"],
            pad=0,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelScale.html
        self.mel_scaler = torchaudio.transforms.MelScale(
            n_mels=params["num_frequencies"],
            sample_rate=params["sample_rate"],
            f_min=params["min_frequency"],
            f_max=params["max_frequency"],
            n_stft=params["n_fft"] // 2 + 1,
            norm=params["mel_scale_norm"],
            mel_scale=params["mel_scale_type"],
        )

    def spectrogram_from_audio(
            self,
            audio: pydub.AudioSegment,
    ) -> np.ndarray:
        """
        Compute a spectrogram from an audio segment.

        Args:
            audio: Audio segment which must match the sample rate of the params

        Returns:
            spectrogram: (channel, frequency, time)
        """
        assert int(audio.frame_rate) == self.p.sample_rate, "Audio sample rate must match params"

        # Get the samples as a numpy array in (batch, samples) shape
        waveform = np.array([c.get_array_of_samples() for c in audio.split_to_mono()])

        # Convert to floats if necessary
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        waveform_tensor = torch.from_numpy(waveform).to(self.device)
        amplitudes_mel = self.mel_amplitudes_from_waveform(waveform_tensor)
        return amplitudes_mel.cpu().numpy()

    def mel_amplitudes_from_waveform(
            self,
            waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-only function to compute Mel-scale amplitudes from a waveform.

        Args:
            waveform: (batch, samples)

        Returns:
            amplitudes_mel: (batch, frequency, time)
        """
        # Compute the complex-valued spectrogram
        spectrogram_complex = self.spectrogram_func(waveform)

        # Take the magnitude
        amplitudes = torch.abs(spectrogram_complex)

        # Convert to mel scale
        return self.mel_scaler(amplitudes)


def spectrogram_from_image(
        image: np.ndarray,
        power: float = 0.25,
        stereo: bool = False,
        max_value: float = 30e6,
) -> np.ndarray:
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
        max_value: float = 30e6,
):
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
        power_for_image=0.25,
        num_frequencies=512,
        sample_rate=44100,
        min_frequency=0,
        max_frequency=10000,
        mel_scale_norm=None,
        mel_scale_type="htk",
        stereo=False,
    ))
    segment = audio_from_spectrogram_image(converter, image)

    savepath = get_savepath(args.savepath, "", ext='.png')
    logger.info(f'saved at : {savepath}')
    cv2.imwrite(savepath, image)

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

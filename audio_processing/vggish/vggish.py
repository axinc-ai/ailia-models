import time
import sys
import argparse
import numpy as np

import ailia  # noqa: E402
import vggish_util as mel_features

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
WAV_PATH = "bus_chatter.wav"

SAVE_WAV_PATH = 'feature.npy'  

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'VGGish',
    WAV_PATH,
    SAVE_WAV_PATH,
)
parser.add_argument(
    '--ailia_audio', action='store_true',
    help='use ailia audio library'
)
args = update_parser(parser)

if args.ailia_audio:
    import soundfile as sf
    import ailia.audio as ailia_audio
else:
    import librosa

# ======================
# Parameters 2 ======================
WEIGHT_PATH = "vggish.onnx"
MODEL_PATH = WEIGHT_PATH + ".prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/vggish/"

NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.

SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

# ======================
# Main function
# ======================

def waveform_to_examples(data, sample_rate, return_tensor=True):
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=SAMPLE_RATE,
        log_offset=LOG_OFFSET,
        window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=NUM_MEL_BINS,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)

    if return_tensor:
        log_mel_examples = log_mel_examples[:, None, :, :].astype("float32")
    return log_mel_examples



def recognize_one_audio(input_path):
    # load audio
    logger.info('Loading wavfile...')

    if args.ailia_audio:
        wav_data,sr = sf.read(input_path)
        wav_data = ailia.audio.resample(wav_data,sr,SAMPLE_RATE)
        samples = wav_data 
        samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    else:
        samples = librosa.load(input_path, sr=SAMPLE_RATE)[0]

    # apply preenphasis filter
    logger.info('Generating input feature...')
    mel = waveform_to_examples(samples, SAMPLE_RATE, True)

    # create instance
    logger.info('Use ailia')
    env_id = args.env_id
    logger.info(f'env_id: {env_id}')
    memory_mode = ailia.get_memory_mode(reuse_interstage=True)
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id, memory_mode=memory_mode)

    # inference
    logger.info('Start inference...')
    if args.benchmark:
        logger.info('BENCHMARK mode')
        for c in range(5) :
            start = int(round(time.time() * 1000))
            result = net.run(mel)
            end = int(round(time.time() * 1000))
            logger.info("\tprocessing time {} ms".format(end-start))
    else:
        result = net.run(mel)

    # save sapareted signal
    savepath = get_savepath(args.savepath, input_path)
    logger.info(f'saved at : {savepath}')

    torch_result = np.load("torch_result.npy")
    error = (torch_result - np.array(result)[0]) ** 2
    RMSE = np.sqrt(np.sum(error)/ np.prod(error.shape)) 
    logger.info('RMSE from original model: '+str(RMSE) )

    np.save(savepath, result)


    logger.info('Saved separated signal. ')
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    for input_file in args.input:
        recognize_one_audio(input_file)

if __name__ == "__main__":
     main()

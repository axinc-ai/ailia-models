import time
import sys

import soundfile as sf
import numpy as np

import ailia

import cv2
import io
import matplotlib.pyplot as plt
#import librosa

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# TODO: FIXME: crnn_audio_classification_util uses torchaudio & torch...


# ======================
# PARAMETERS
# ======================
# https://freesound.org/people/www.bonson.ca/sounds/24965/
WAVE_PATH = "24965__www-bonson-ca__bigdogbarking-02.wav"

# WAVE_PATH="dog.wav" # dog_bark 0.5050086379051208

WEIGHT_PATH = "crnn_audio_classification.onnx"
MODEL_PATH = "crnn_audio_classification.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/crnn_audio_classification/"

SAMPLING_RATE = 16000
WIN_LENGTH = int(SAMPLING_RATE * 0.02)
HOP_LENGTH = int(SAMPLING_RATE * 0.01)


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'CRNN Audio Classification.', WAVE_PATH, None, input_ftype='audio')
parser.add_argument(
    '--ailia_audio', action='store_true',
    help='use ailia audio library'
)
# overwrite
parser.add_argument(
    '-i', '--input', metavar='WAV',
    default=WAVE_PATH,
    help='The input wav path.',
)
parser.add_argument(
    '-v',
    action='store_true',
    help='use microphone input',
)
args = update_parser(parser)

if args.ailia_audio:
    from crnn_audio_classification_util_ailia import MelspectrogramStretch
else:
    from crnn_audio_classification_util import MelspectrogramStretch  # noqa: E402

if args.v:
    import pyaudio
    # pyaudio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RECODING_SAMPING_RATE = 48000
    THRESHOLD = 0.02

# ======================
# Postprocess
# ======================
def postprocess(x):
    classes = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
        'street_music'
    ]
    out = np.exp(x)
    max_ind = out.argmax().item()
    return classes[max_ind], out[:, max_ind].item()


# ======================
# Sound Utils
# ======================
def record_microphone_input():
    logger.info('Ready...')
    time.sleep(1)
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RECODING_SAMPING_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    time.sleep(1)
    logger.info("Please speak something")

    frames = []
    count_uv = 0

    stream.start_stream()
    while True:
        if len(frames) > 500000:
            break
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16) / 32768.0
        if data.max() > THRESHOLD:
            frames.extend(data)
            count_uv = 0
        elif len(frames) > 0:
            count_uv += 1
            if count_uv > 48:
                break
            frames.extend(data)

    # logger.info("Translating")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wav = np.array(frames)
    if args.ailia_audio:
        return ailia.audio.resample(wav, RECODING_SAMPING_RATE, SAMPLING_RATE)
    else:
        return librosa.resample(wav, RECODING_SAMPING_RATE, SAMPLING_RATE)


def create_spectrogram(wav):
    if args.ailia_audio:
        spectrogram = ailia.audio.create_spectrogram(
            wav,
            fft_n=WIN_LENGTH,
            hop_n=HOP_LENGTH,
            win_n=WIN_LENGTH,
            win_type="hamming",
        )
        spec_length = np.array(([spectrogram.shape[1]-1]))
    else:
        stft = librosa.stft(
            wav,
            n_fft=WIN_LENGTH,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            window='hamming',
        )
        stft, _ = librosa.magphase(stft)
        spectrogram = np.log1p(stft)
        spec_length = np.array(([stft.shape[1]-1]))

        mean = spectrogram.mean()
        std = spectrogram.std()
        spectrogram -= mean
        spectrogram /= std

    spectrogram = np.log1p(spectrogram)

    return (spectrogram, spec_length)


# ======================
# Main function
# ======================
def crnn(data, session, fixed_shape = False):
    # normal inference
    spec = MelspectrogramStretch()
    xt, lengths = spec.forward(data)

    # inference
    lengths_np = np.zeros((1))
    lengths_np[0] = lengths[0]

    # fixed shape
    if fixed_shape:
        new_xt = np.zeros((1, 1, 128, 176))
        new_xt[:, :, :xt.shape[2], :xt.shape[3]] = xt
        xt = new_xt
        lengths_np = np.array([176.])

    results = session.predict({"data": xt, "lengths": lengths_np})

    label, conf = postprocess(results[0])

    return label, conf


def microphone_input_recognition():
    try:
        print('processing...')
        session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
        frame_shown = False
        while True:
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
                break

            wav = record_microphone_input()

            # create instance

            data = (wav, SAMPLING_RATE)
            label, conf = crnn(data, session, fixed_shape = True)

            plt.specgram(wav,Fs=1)
            plt.title('Predicted class is = {}, confidence = {}'.format(label, conf))

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)

            cv2.imshow('frame', img)
            frame_shown = True
            time.sleep(0)

            logger.info(label)
            logger.info(conf)

    except KeyboardInterrupt:
        logger.info('script finished successfully.')


def wavfile_input_recognition():
    # load audio
    for input_data_path in args.input:
        logger.info('=' * 80)
        logger.info(f'input: {input_data_path}')
        data = sf.read(input_data_path)

        # create instance
        session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for c in range(5):
                start = int(round(time.time() * 1000))
                label, conf = crnn(data, session)
                end = int(round(time.time() * 1000))
                logger.info("\tailia processing time {} ms".format(end-start))
        else:
            label, conf = crnn(data, session)

        logger.info(label)
        logger.info(conf)

        logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # microphone input mode
    if args.v:
        microphone_input_recognition()
    # sound file input mode
    else:
        wavfile_input_recognition()


if __name__ == "__main__":
    main()

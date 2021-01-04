import time
import sys

import soundfile as sf
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from crnn_audio_classification_util import MelspectrogramStretch  # noqa: E402


# ======================
# PARAMETERS
# ======================
# https://freesound.org/people/www.bonson.ca/sounds/24965/
WAVE_PATH = "24965__www-bonson-ca__bigdogbarking-02.wav"

# WAVE_PATH="dog.wav" # dog_bark 0.5050086379051208

WEIGHT_PATH = "crnn_audio_classification.onnx"
MODEL_PATH = "crnn_audio_classification.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/crnn_audio_classification/"


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('CRNN Audio Classification.', WAVE_PATH, None)
args = update_parser(parser)


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
# Main function
# ======================
def crnn(data, session):
    # normal inference
    spec = MelspectrogramStretch()
    xt, lengths = spec.forward(data)

    # inference
    lengths_np = np.zeros((1))
    lengths_np[0] = lengths[0]
    results = session.predict({"data": xt, "lengths": lengths_np})

    label, conf = postprocess(results[0])

    return label, conf


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load audio
    data = sf.read(args.input)

    # create instance
    session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))
            label, conf = crnn(data, session)
            end = int(round(time.time() * 1000))
            print("\tailia processing time {} ms".format(end-start))
    else:
        label, conf = crnn(data, session)

    print(label)
    print(conf)

    print('Script finished successfully.')


if __name__ == "__main__":
    main()

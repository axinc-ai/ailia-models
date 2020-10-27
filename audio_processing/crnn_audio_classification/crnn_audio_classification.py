import time
import sys
import argparse

import numpy as np

import ailia  # noqa: E402
import onnxruntime

import soundfile as sf

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from crnn_audio_classification_util import MelspectrogramStretch  # noqa: E402

# ======================
# Arguemnt Parser Config
# ======================

WAVE_PATH="24965__www-bonson-ca__bigdogbarking-02.wav" #https://freesound.org/people/www.bonson.ca/sounds/24965/
#WAVE_PATH="dog.wav" #dog_bark 0.5050086379051208

parser = argparse.ArgumentParser(
    description='CRNN Audio Classification.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=WAVE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)
parser.add_argument(
    '-o', '--onnx',
    action='store_true',
    help='Running on onnx runtime'
)

args = parser.parse_args()


# ======================
# PARAMETERS
# ======================

WEIGHT_PATH = "crnn_audio_classification.onnx"
MODEL_PATH = "crnn_audio_classification.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/crnn_audio_classification/"

# ======================
# Postprocess
# ======================


def postprocess(x):
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    out = np.exp(x)
    max_ind = out.argmax().item()      
    return classes[max_ind], out[:,max_ind].item()


# ======================
# Main function
# ======================

def crnn(data, session):
    # normal inference
    spec = MelspectrogramStretch()
    xt, lengths = spec.forward(data)

    # inference
    if args.onnx:
        results = session.run(["conf"],{ "data": xt, "lengths": lengths})
    else:
        results = net.predict({ "data": xt, "lengths": lengths})

    label, conf = postprocess(results[0])

    return label, conf

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load audio
    data = sf.read(args.input)

    # create instance
    if args.onnx:
        session = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        env_id = ailia.get_gpu_environment_id()
        print(f'env_id: {env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

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

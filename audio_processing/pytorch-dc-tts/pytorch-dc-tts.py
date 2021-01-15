import time
import sys
import argparse

import numpy as np

import ailia  # noqa: E402
from utils import get_test_data, save_to_wav

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402


# ======================
# PARAMETERS
# ======================
SENTENCE = 'The boy was there when the sun rose.'

SAVE_WAV_PATH = 'output.wav'

WEIGHT_PATH_T2M = 'text2mel.onnx'
MODEL_PATH_T2M = 'text2mel.onnx.prototxt'
REMOTE_PATH_T2M = 'https://storage.googleapis.com/ailia-models/pytorch-dc-tts/'

WEIGHT_PATH_SSRM = 'ssrn.onnx'
MODEL_PATH_SSRM = 'ssrn.onnx.prototxt'
REMOTE_PATH_SSRM = 'https://storage.googleapis.com/ailia-models/pytorch-dc-tts/'

MAX_T = 210

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description=' Efficiently Trainable Text-to-Speech System Based on' +
    'Deep Convolutional Networks with Guided Attention'
)
parser.add_argument(
    '-i', '--input', metavar='INPUT_SENTENCE',
    default=SENTENCE,
    help='The input sentence.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_WAV_PATH',
    default=SAVE_WAV_PATH,
    help='Save path for the output audio.'
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
    default=False,
    help='Use onnx runtime'
)
args = parser.parse_args()


# ======================
# Utils
# ======================


# ======================
# Main function
# ======================
def preprocess(SENTENCE):
    L = get_test_data([SENTENCE], len(SENTENCE))
    zeros = np.zeros((1, 80, 1), np.float32)
    Y = zeros
    A = None
    return L, Y, zeros, A


def inference(net_t2m, net_ssrm, L, Y, zeros, A):
    Y = inference_by_text2mel(net_t2m, L, Y, zeros, A)
    Z = inference_by_ssr(net_ssrm, Y)
    return Z[0, :, :].T


def inference_by_text2mel(net_t2m, L, Y, zeros, A):
    for t in (range(MAX_T)):
        if not args.onnx:
            net_t2m.set_input_blob_shape(Y.shape, net_t2m.find_blob_index_by_name('input.2'))
            _, Y_t, A = net_t2m.predict({'input.1':L, 'input.2':Y})
        else:
            first_input_name = net_t2m.get_inputs()[0].name
            second_input_name = net_t2m.get_inputs()[1].name
            first_output_name = net_t2m.get_outputs()[0].name
            second_output_name = net_t2m.get_outputs()[1].name
            third_output_name = net_t2m.get_outputs()[2].name

            _, Y_t, A = net_t2m.run(
                [first_output_name, second_output_name, third_output_name], 
                {first_input_name: L, second_input_name: Y}
            )

        Y = np.concatenate([zeros, Y_t], 2)
        attention = np.argmax(A[0, :, -1], 0)
        vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    return Y


def inference_by_ssr(net_ssrm, Y):
    if not args.onnx:
        _, Z = net_ssrm.predict({'input.1':Y})
    else:
        first_input_name = net_ssrm.get_inputs()[0].name
        first_output_name = net_ssrm.get_outputs()[0].name
        second_output_name = net_ssrm.get_outputs()[1].name

        _, Z = net_ssrm.run(
            [first_output_name, second_output_name], 
            {first_input_name: Y}
        )

    return Z


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH_T2M, MODEL_PATH_T2M, REMOTE_PATH_T2M)
    check_and_download_models(WEIGHT_PATH_SSRM, MODEL_PATH_SSRM, REMOTE_PATH_SSRM)

    # prepare data
    L, Y, zeros, A = preprocess(SENTENCE)

    # model initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    if not args.onnx:
        net_t2m = ailia.Net(MODEL_PATH_T2M, WEIGHT_PATH_T2M, env_id=env_id)
        net_ssrm = ailia.Net(MODEL_PATH_SSRM, WEIGHT_PATH_SSRM, env_id=env_id)
    else:
        print('Let us try by onnxruntime')
        import onnxruntime
        net_t2m = onnxruntime.InferenceSession(WEIGHT_PATH_T2M)
        net_ssrm = onnxruntime.InferenceSession(WEIGHT_PATH_SSRM)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for c in range(5):
            start = int(round(time.time() * 1000))
            out = inference(net_t2m, net_ssrm, L, Y, zeros, A)
            end = int(round(time.time() * 1000))
            print("\tailia processing time {} ms".format(end-start))
    else:
        out = inference(net_t2m, net_ssrm, L, Y, zeros, A)

    save_to_wav(out, SAVE_WAV_PATH)
    print('Script finished successfully.')


if __name__ == "__main__":
    main()

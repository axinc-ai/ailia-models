import sys
import time
import codecs
import argparse
import numpy

import cv2

import ailia
# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402

import string

# ======================
# PARAMETERS
# ======================
MODEL_PATH = 'None-ResNet-None-CTC.onnx.prototxt'
WEIGHT_PATH = 'None-ResNet-None-CTC.onnx'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/deep-text-recognition-benchmark/'

IMAGE_FOLDER_PATH = 'demo_image/'
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 32


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='deep text recognition benchmark.'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE_FOLDER_PATH',
    default=IMAGE_FOLDER_PATH,
    help='The input image folder path.'
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
    help='Use onnx runtime'
)
args = parser.parse_args()


# ======================
# Utils
# ======================

def ctc_decode(text_index, length, character):
    dict_character = list(character)

    dict = {}
    for i, char in enumerate(dict_character):
        dict[char] = i + 1

    character = ['[CTCblank]'] + dict_character
    CTC_BLANK = 0
    
    texts = []
    for index, l in enumerate(length):
        t = text_index[index, :]

        char_list = []
        for i in range(l):
            if t[i] != CTC_BLANK and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(character[t[i]])
        text = ''.join(char_list)

        texts.append(text)

    return texts

def preprocess_image(sample):
    sample = cv2.resize(sample,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_CUBIC)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sample = sample/127.5 - 1.0
    return sample

def softmax(x):
    u = numpy.sum(numpy.exp(x))
    return numpy.exp(x)/u

dashed_line = '-' * 80

def recognize_from_image():
    import glob
    image_path_list = sorted(glob.glob(args.input+"/*"))
    #image_path_list = [image_path_list[0]]

    head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
    
    print(f'{dashed_line}\n{head}\n{dashed_line}')

    if args.onnx:
        import onnxruntime
        session = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        env_id = ailia.get_gpu_environment_id()
        session = ailia.Net(MODEL_PATH,WEIGHT_PATH,env_id=env_id)

    for path in image_path_list:
        recognize_one_image(path,session)

def recognize_one_image(image_path,session):
    """ model configuration """
    character = '0123456789abcdefghijklmnopqrstuvwxyz'
    imgH = IMAGE_HEIGHT
    imgW = IMAGE_WIDTH
    PAD = False
    batch_size = 1
    workers = 4
    batch_max_length = 25
    time_size = 26

    # predict
    input_img = numpy.zeros((batch_size,1,imgH,imgW),dtype=numpy.float32)
    
    sample = cv2.imread(image_path)
    sample = preprocess_image(sample)
    input_img[0,:,:,:] = sample

    if args.onnx:
        session.get_modelmeta()
        first_input_name = session.get_inputs()[0].name
        preds = session.run([], {first_input_name: input_img})
        preds = preds[0]
    else:
        preds = session.predict(input_img)

    # Select max probabilty (greedy decoding) then decode index to character
    preds_size = [int(preds.shape[1])] * batch_size
    preds_index = numpy.argmax(preds,axis=2)

    preds_str = ctc_decode(preds_index, preds_size, character)
    preds_prob = numpy.zeros(preds.shape)
    
    for b in range(0,batch_size):
        for t in range(0,time_size):
            preds_prob[b,t,:]=softmax(preds[b,t,:])

    preds_max_prob = numpy.max(preds_prob,axis=2)
    for img_name, pred, pred_max_prob in zip([image_path], preds_str, preds_max_prob):
        confidence_score = pred_max_prob.cumprod(axis=0)[-1]

        print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')


if __name__ == '__main__':
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    recognize_from_image()

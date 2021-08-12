import sys, os
import time
import copy
import json
from logging import getLogger
import numpy as np
import ailia
import glob
from tqdm import tqdm

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

import chainer
from chainer.backends import cuda
from PIL import Image

logger = getLogger(__name__)



# ======================
# Parameters
# ======================

#WEIGHT_PATH = 'spade_pix2pix.onnx'
WEIGHT_PATH = 'out_spade_pix2pix.onnx'
MODEL_PATH = 'spade_pix2pix.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/spade_pix2pix/'
IN_IMAGE_PATH = './inputs/'
OUT_IMAGE_PATH = './outputs/'



# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Invertible Denoising Network', IN_IMAGE_PATH, OUT_IMAGE_PATH
)
parser.add_argument(
    '-n', '--onnx',
    action='store_true',
    default=False,
    help='Use onnxruntime'
)
args = update_parser(parser)



# ======================
# Util functions
# ======================

#This parameter dependent on RGB-combination of label
CLASS_COLOR = [[1, 0, 0], #eye
               [0, 1, 0], #face
               [0, 0, 1], #hair
               [1, 0, 1], #other
               [1, 1, 0]] #background
BACKGROUND_INDEX = 4

#the range of RGB is from zero to one.
def label2onehot(label, threshold=0.4, skip_bg=False, dtype='uint8'):
    label = label > threshold

    onehot = None
    xp = cuda.get_array_module(label)
    na = xp.newaxis

    for i in range(len(CLASS_COLOR)):
        if skip_bg and i == BACKGROUND_INDEX:
            continue

        detecter = xp.array(CLASS_COLOR[i], dtype=dtype)[:, na, na]
        detecter = detecter.repeat(label.shape[1], axis=1)
        detecter = detecter.repeat(label.shape[2], axis=2)

        mask = xp.sum(label == detecter,
            axis=0, keepdims=True, dtype=dtype) == 3

        if i == 0:
            onehot = mask
        else:
            onehot = xp.concatenate((onehot, mask), axis=0)

    return onehot



# ======================
# Main functions
# ======================

def main():
    # model files check and download
    #check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)

    import onnxruntime
    net = onnxruntime.InferenceSession(WEIGHT_PATH)

    os.makedirs(IN_IMAGE_PATH, exist_ok=True)
    os.makedirs(OUT_IMAGE_PATH, exist_ok=True)

    files = glob.glob(IN_IMAGE_PATH + '*.png')
    if len(files) == 0:
        print('Erorr: No files to load in \'' + IN_IMAGE_PATH + '\'.')
        return

    num = 0
    for filename in files:
        print(filename + ': ', end="")
        src_img = Image.open(filename).convert('RGB')
        if src_img is None:
            print('Not Loaded')
            continue

        print('Loaded')
        src_array = np.array(src_img, dtype='float32')
        src_array = src_array.transpose((2, 0, 1)) / 255

        x_array = src_array[:3, :, :256]
        c_array = src_array[:3, :, 256:512]

        x_onehot = label2onehot(x_array, threshold=0.4, skip_bg=True, dtype='float32')
        #x = chainer.Variable(x_onehot[np.newaxis, :, :, :].astype('float32'))

        c_array = c_array * x_onehot[2]  # crop with hair label
        #c = chainer.Variable(c_array[np.newaxis, :, :, :].astype('float32'))

        #np.set_printoptions(threshold=np.inf)
        #print(c_array[np.newaxis, :, :, :].astype('float32'))
        #exit()

        #output = net.predict([
        #    x_onehot[np.newaxis, :, :, :].astype('float32'),
        #    c_array[np.newaxis, :, :, :].astype('float32')
        #])

        output = net.run(None, {
            net.get_inputs()[0].name: x_onehot[np.newaxis, :, :, :].astype('float32'),
            net.get_inputs()[1].name: c_array[np.newaxis, :, :, :].astype('float32')
        })

        #np.set_printoptions(threshold=np.inf)
        #print(output)
        #exit()

        np.set_printoptions(threshold=np.inf)
        output = output[0]

        print(output)
        exit()

        x_array = np.transpose(x_array, (1, 2, 0))
        out_array = np.transpose((output.squeeze(0) + 1) / 2, (1, 2, 0))

        img_array = np.concatenate((x_array, out_array), axis=1) * 255
        #img_array = np.transpose(output.squeeze(0), (1, 2, 0)) * 255
        img = Image.fromarray(img_array.astype('uint8'))

        path = OUT_IMAGE_PATH + str(num) + '.png'
        img.save(path)

        num += 1


if __name__ == '__main__':
    main()

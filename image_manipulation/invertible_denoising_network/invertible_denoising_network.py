import sys, os
import time
import copy
import json
from logging import getLogger
import numpy as np
import cv2
import ailia
import glob
import torch
from tqdm import tqdm

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

logger = getLogger(__name__)



# ======================
# Parameters
# ======================

WEIGHT_PATH = 'InvDN.onnx'
MODEL_PATH = 'InvDN.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/invertible_denoising_network/'
IMAGE_PATH = './input_images/'
SAVE_IMAGE_PATH = './output_images/'



# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Invertible Denoising Network', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-n', '--onnx',
    action='store_true',
    default=False,
    help='Use onnxruntime'
)
args = update_parser(parser)



# ======================
# Main functions
# ======================

def tensor2img_Real(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        # n_img = len(tensor)
        # img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = tensor.numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def read_img_array(img):
    '''read image array and preprocess
    return: Numpy float32, HWC, BGR, [0,1]'''
    img = img.astype(np.float32) / 255.
    return img

def predict(input):
    # net initialize, predict
    if args.onnx:
        import onnxruntime
        sess = onnxruntime.InferenceSession(WEIGHT_PATH)
        inputs = {
            sess.get_inputs()[0].name: input.astype(np.float32),
            sess.get_inputs()[1].name: np.array([1], dtype=np.int32)
        }
        preds = sess.run(None, inputs)
    else:
        print('Sample for ailia is not implemented.')
        exit()
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
        preds = net.predict({
            'input': input.astype(np.float32),
            'gaussian_scale': np.array([1])
        })

    return preds[1]

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    inputs = glob.glob("./input_images/*.PNG")
    for input in tqdm(inputs):
        basename = os.path.basename(input)

        img = cv2.imread(input)
        img = read_img_array(img)
        img = np.expand_dims(img, 0)
        img = np.transpose(img, (0, 3, 1, 2))

        output = predict(img)

        output = output[:, :3, :, :]
        output = torch.from_numpy(output.astype(np.float32)).clone()
        output = tensor2img_Real(output)
        output = np.transpose(output, (1, 2, 0))

        save_file = os.path.join('./output_images', basename)
        cv2.imwrite(save_file, output)

if __name__ == '__main__':
    main()

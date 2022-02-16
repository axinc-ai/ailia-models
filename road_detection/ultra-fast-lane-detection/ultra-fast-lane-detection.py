import sys
import time
import numpy as np
import glob
import cv2

import onnxruntime
import ailia

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

# ======================
# Parameters 1
# ======================

SAVE_IMAGE_PATH = 'output.jpg'

INPUT_IMAGE_PATH = 'input.jpg'
# INPUT_IMAGE_PATH = '02370.jpg'

MODEL_LISTS = ['culane','tusimple']

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'lane_detection',
    INPUT_IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-i','--input', type=str,
    default=INPUT_IMAGE_PATH,
    help='The input image for input image.'
)
parser.add_argument(
    '-v', '--video', type=str,
    # default=STEREO_DATA,
    help='The input video for pole detection.'
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='culane', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
args = update_parser(parser)

# # ======================
# # Select model
# # ======================

if args.arch=="culane":
    WEIGHT_PATH = 'culane_18.onnx'
    MODEL_PATH = 'culane_18.onnx.prototxt'
else:
    WEIGHT_PATH = 'tusimple_18.onnx'
    MODEL_PATH = 'tusimple_18.onnx.prototxt'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/ultra-fast-lane-detection/'

# # ======================
# # Main functions
# # ======================

# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.406, 0.456, 0.485]) # this is correct
std = np.array([0.225, 0.224, 0.229])

def preprocessing(img):

    input_img = cv2.resize(img,(800,288))
    
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img / 255.0
    input_img = input_img - mean / std    
    input_img = input_img.transpose(2, 0, 1)
    
    return np.expand_dims(input_img, 0).astype(np.float32)

import scipy.special, tqdm

griding_num = 200
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
cls_num_per_lane = 18
def recognize_from_image(net):
    # input image loop
    for image_path in args.input:
        img = cv2.imread(image_path)

        input_name = 'input.1'
        output_name = '200'

        input_tensor = preprocessing(img)

        if args.onnx:
            y = net.run( [output_name], { input_name : input_tensor })[0][0]
        else:
            y = net.run(input_tensor)[0][0]

        col_sample = np.linspace(0, 800 - 1, griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor

        print(np.array(y).shape)
        out_j = y
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

        idx = np.arange(griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        print(idx.shape)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == griding_num] = 0
        out_j = loc

        print(out_j.shape)

        # import pdb; pdb.set_trace()
        vis = cv2.imread(args.input[0])
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(vis,ppp,5,(0,255,0),-1)

        
        # print(y.shape)
        save_img = cv2.resize( vis , ( int(vis.shape[1] / 4 ) , int(vis.shape[0] / 4 )))

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, save_img)

    logger.info('Script finished successfully.')

def recognize_from_video(net):

    capture = webcamera_utils.get_capture(args.video)

    input_name = 'input.1'
    output_name = '200'

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        print(frame.shape)

        input_img = frame
        input_tensor = preprocessing(input_img)

   
        if args.onnx:
            y = net.run( [output_name], { input_name : input_tensor })[0][0]
        else:
            disparity_map = model.run(imgs)[0]

        print(y)

        col_sample = np.linspace(0, 800 - 1, griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor

        # print(np.array(y).shape)
        out_j = y
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

        idx = np.arange(griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == griding_num] = 0
        out_j = loc

        # import pdb; pdb.set_trace()
        vis = frame
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(vis,ppp,5,(0,255,0),-1)

    
        cv2.imshow('output', vis)
        cv2.waitKey(3)

    capture.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')
  
def main():

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # net initialize
    if args.onnx:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)
    else:
        logger.info(f'env_id: {args.env_id}')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id)
    

    if args.video is not None:
        # video mode
        recognize_from_video(net)
    else:
        # image mode
        recognize_from_image(net)

if __name__ == '__main__':
    main()

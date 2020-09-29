import time
import math
import sys
import argparse
import pathlib

import torch
import numpy as np
import cv2

import ailia


# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models 
from webcamera_utils import adjust_frame_size  

# ======================
# Parameters
# ======================

IMAGE_PATH = 'road.jpg'
SAVE_IMAGE_PATH = 'output_ddd.png'

THRESHOLD = 0.4 #Threshold for filteing for filtering (from 0.0 to 1.0)
K_VALUE = 40    #K value for topK function

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='CenterNet model'
)
parser.add_argument(
    '-i', '--input', metavar='IMAGE',
    default=IMAGE_PATH,
    help='The input image path.'
)
parser.add_argument(
    '-v', '--video', metavar='VIDEO',
    default=None,
    help='The input video path. ' +
         'If the VIDEO argument is set to 0, the webcam input will be used.'
)
parser.add_argument(
    '-s', '--savepath', metavar='SAVE_IMAGE_PATH',
    default=SAVE_IMAGE_PATH,
    help='Save path for the output image.'
)
parser.add_argument(
    '-b', '--benchmark',
    action='store_true',
    help='Running the inference on the same input 5 times ' +
         'to measure execution performance. (Cannot be used in video mode)'
)

args = parser.parse_args()

WEIGHT_PATH = './ddd_dlav0.onnx'
MODEL_PATH = WEIGHT_PATH + '.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/centernet/'

# ======================
# Main functions
# ======================

from ddd_files.ddd import ddd_decode, pre_process, post_process, sigmoid
from ddd_files.ddd_plot import add_3d_detection

    
HEADS = ['hm', 'dep', 'rot', 'dim', 'wh', 'reg']
def process(image, net):
    # reformat list output into dictionary with original labels
    net.predict(image)
    output_raw = net.get_results()
    output = {label: out for label, out in zip(HEADS, output_raw)}
 
    output['hm'] = sigmoid(output['hm'])
    output['dep'] = 1. / (sigmoid(output['dep']) + 1e-6) - 1.
    
    dets = ddd_decode(output['hm'][0], output['rot'][0], output['dep'][0],
                      output['dim'][0], wh=output['wh'][0], reg=output['reg'][0], k=K_VALUE)
    return output, dets

def detect_objects(img, net):
    pre_img, meta = pre_process(img)
    outputs, dets = process(pre_img, net)
    dets = post_process(dets, meta, THRESHOLD)
    return dets, meta['calib']
    
def recognize_from_image(filename, net):
   
    img = cv2.imread(filename)

    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            dets, calib = detect_objects(img, net)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        dets, calib = detect_objects(img, net)
   
    out_img = add_3d_detection(img, dets, calib.astype(np.float32))
  
    cv2.imwrite(args.savepath, out_img)
    print('Script finished successfully.')
    
    #cv2.imshow('demo', im2show)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()
    
def recognize_from_video(video, net):
    if video == '0':
        print('[INFO] Webcam mode is activated')
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("[ERROR] webcamera not found")
            sys.exit(1)
    else:
        if pathlib.Path(video).exists():
            capture = cv2.VideoCapture(video)

    while(True):
        ret, img = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        dets, calib = detect_objects(img, net)
        out_img = add_3d_detection(img, dets, calib)
        cv2.imshow('frame', out_img)
        
        # press q to end video capture
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        if not ret:
            continue

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')
    
def main():
    # load model
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH,env_id=env_id)
 
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video, net)
    else:
        # image mode
        recognize_from_image(args.input, net)


if __name__=='__main__':
    main()
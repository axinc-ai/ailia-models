import time
import sys
import argparse
import pathlib

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import ailia
import dbface_utils

# import original modules
sys.path.append('../util')
from model_utils import check_and_download_models 
from detector_utils import load_image 

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'dbface_pytorch.onnx'
MODEL_PATH = 'dbface_pytorch.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/dbface/'

IMAGE_PATH = 'selfie.png'
SAVE_IMAGE_PATH = 'selfie_output.png'

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='DBFace model'
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

# ======================
# Secondaty Functions
# ======================

def nms(objs, iou):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def preprocess(img):
    img = dbface_utils.pad(img)
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    img = ((img / 255.0 - mean) / std).astype(np.float32)
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, 0)
    return img


def detect_objects(img):
    img = preprocess(img)
    
    env_id = ailia.get_gpu_environment_id()
    detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    detector.set_input_shape((1, 3, img.shape[2], img.shape[3]))
    hm, box, landmark = detector.predict({'input.1': img})
    
    hm_pool = F.max_pool2d(torch.from_numpy(hm), 3, 1, 1)
    scores, indices = ((torch.from_numpy(hm) == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]
    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices // hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = torch.from_numpy(box).cpu().squeeze().data.numpy()
    landmark = torch.from_numpy(landmark).cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < THRESHOLD:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (dbface_utils.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(dbface_utils.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    
    return nms(objs, iou=IOU)


# ======================
# Main functions
# ======================
def recognize_from_image(filename):
    # load input image
    img = load_image(filename)
    print(f'input image shape: {img.shape}')
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            objs = detect_objects(img)
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        objs = detect_objects(img)
        
    # show image 
    for obj in objs:
        dbface_utils.drawbbox(img, obj)
    cv2.imwrite(args.savepath, img)

    print('Script finished successfully.')


def recognize_from_video(video): 
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
        objs = detect_objects(img)
        for obj in objs:
            dbface_utils.drawbbox(img, obj)
        cv2.imshow('frame', img)

        # press q to end video capture
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        if not ret:
            continue

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(args.video)
    else:
        # image mode
        recognize_from_image(args.input)


if __name__=='__main__':
    main()

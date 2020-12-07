import os
import sys
import time
import argparse
from glob import glob

import cv2
import numpy as np

import ailia

# import original modules
sys.path.append('../../util')
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import adjust_frame_size, get_capture  # noqa: E402C


# ======================
# Parameters
# ======================
WEIGHT_PATH = 'u2net_portrait.onnx'
MODEL_PATH = 'u2net_portrait.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net_portrait/'

IMAGE_PATH = 'your_portrait_im/kid1.jpg'
SAVE_IMAGE_PATH = 'your_portrait_results/kid1.jpg'

FACE_CASCADE_MODEL_PATH = 'haarcascade_frontalface_default.xml'
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512


# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection'
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
# Utils
# ======================
def detect_single_face(face_cascade,img):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if(len(faces)==0):
        print("Warming: no face detection, the portrait u2net will run on the whole image!")
        return None

    # filter to keep the largest face
    wh = 0
    idx = 0
    for i in range(0, len(faces)):
        (x,y,w,h) = faces[i]
        if (wh<w*h):
            idx = i
            wh = w*h

    return faces[idx]


# crop, pad and resize face region to 512x512 resolution
def crop_face(img, face):

    # no face detected, return the whole image and the inference will run on the whole image
    if(face is None):
        return cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
    (x, y, w, h) = face

    height, width = img.shape[0:2]

    # crop the face with a bigger bbox
    hmw = h - w

    l,r,t,b = 0,0,0,0
    lpad = int(float(w)*0.4)
    left = x-lpad
    if(left<0):
        l = lpad-x
        left = 0

    rpad = int(float(w)*0.4)
    right = x+w+rpad
    if(right>width):
        r = right-width
        right = width

    tpad = int(float(h)*0.6)
    top = y - tpad
    if(top<0):
        t = tpad-y
        top = 0

    bpad  = int(float(h)*0.2)
    bottom = y+h+bpad
    if(bottom>height):
        b = bottom-height
        bottom = height


    im_face = img[top:bottom,left:right]
    if(len(im_face.shape)==2):
        im_face = np.repeat(im_face[:,:,np.newaxis],(1,1,3))

    im_face = np.pad(im_face,((t,b),(l,r),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))

    # pad to achieve image with square shape for avoding face deformation after resizing
    hf,wf = im_face.shape[0:2]
    if(hf-2>wf):
        wfp = int((hf-wf)/2)
        im_face = np.pad(im_face,((0,0),(wfp,wfp),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))
    elif(wf-2>hf):
        hfp = int((wf-hf)/2)
        im_face = np.pad(im_face,((hfp,hfp),(0,0),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))

    # resize to have 512x512 resolution
    im_face = cv2.resize(im_face, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)

    return im_face


def preprocess(img):
    # Load the cascade face detection model
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_MODEL_PATH)

    face = detect_single_face(face_cascade, img)
    im_face = crop_face(img, face)

    # normalize the input
    input_img = np.zeros((im_face.shape[0],im_face.shape[1],3))
    im_face = im_face/np.max(im_face)
    input_img[:,:,0] = (im_face[:,:,2]-0.406)/0.225
    input_img[:,:,1] = (im_face[:,:,1]-0.456)/0.224
    input_img[:,:,2] = (im_face[:,:,0]-0.485)/0.229

    # convert BGR to RGB
    input_img = input_img.transpose((2, 0, 1))
    input_img = input_img[np.newaxis,:,:,:]

    return input_img


def post_process(d1):
    pred = 1.0 - d1[:,0,:,:]

    # normalization
    ma = np.max(pred)
    mi = np.min(pred)
    pred = (pred-mi)/(ma-mi)

    return (np.squeeze(pred)*255).astype(np.uint8)


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    img = cv2.imread(IMAGE_PATH)
    print(f'input image shape: {img.shape}')
    input_img = preprocess(img)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            d1,d2,d3,d4,d5,d6,d7 = net.predict({'input.1': input_img})
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        d1,d2,d3,d4,d5,d6,d7 = net.predict({'input.1': input_img})

    out_img = post_process(d1)
    cv2.imwrite(args.savepath, out_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    capture = get_capture(args.video)
    while(True):
        ret, img = capture.read()
        
        # press q to end video capture
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_img = preprocess(img)
        print(input_img.shape)
        d1,d2,d3,d4,d5,d6,d7 = net.predict({'input.1': input_img})
        out_img = post_process(d1)
        cv2.imshow('frame', out_img)

    capture.release()
    cv2.destroyAllWindows()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()

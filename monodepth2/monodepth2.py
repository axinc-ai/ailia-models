import time
import os
import urllib.request

import PIL.Image as pil
import cv2
import matplotlib.pyplot as plt
import numpy as np

import ailia


# TODO Video mode
IMG_PATH = 'input.jpg'
WIDTH = 640
HEIGHT = 192 

MODEL_NAME = 'monodepth2_mono+stereo_640x192'

ENC_WEIGHT_PATH = MODEL_NAME + '_enc.onnx'
ENC_MODEL_PATH = MODEL_NAME + '_enc.onnx.prototxt'
DEC_WEIGHT_PATH = MODEL_NAME + '_dec.onnx'
DEC_MODEL_PATH = MODEL_NAME + '_dec.onnx.prototxt'

RMT_CKPT = "https://storage.googleapis.com/ailia-models/monodepth2/"


if not os.path.exists(ENC_MODEL_PATH):
    print('enocder model downloading...')
    urllib.request.urlretrieve(RMT_CKPT + ENC_MODEL_PATH, ENC_MODEL_PATH)
if not os.path.exists(ENC_WEIGHT_PATH):
    urllib.request.urlretrieve(RMT_CKPT + ENC_WEIGHT_PATH, ENC_WEIGHT_PATH)

if not os.path.exists(DEC_MODEL_PATH):
    print('decoder model downloading...')
    urllib.request.urlretrieve(RMT_CKPT + DEC_MODEL_PATH, DEC_MODEL_PATH)
if not os.path.exists(DEC_WEIGHT_PATH):
    urllib.request.urlretrieve(RMT_CKPT + DEC_WEIGHT_PATH, DEC_WEIGHT_PATH)


def load_image(fname, width, height):
    img = pil.open(fname).convert('RGB')
    original_width, original_height = img.size
    img = np.asarray(img.resize((width, height), pil.LANCZOS))
    # [x, y, channel] --> [1, channel, x, y]
    imgs = np.expand_dims(np.rollaxis(img, 2, 0), axis=0).astype(np.float32)
    assert imgs.shape[1] == 3
    assert imgs.shape[2] == height
    assert imgs.shape[3] == width
    imgs = imgs / 255.0
    return imgs, original_width, original_height


def result_plot(disp, original_width, original_height):
    disp = disp.squeeze()
    disp_resized = cv2.resize(
        disp,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )
    vmax = np.percentile(disp_resized, 95)
    plt.imsave('output.png', disp_resized, cmap='magma', vmax=vmax)


def recognize_from_image(enc_net, dec_net, img_path):
    # Data loading
    input_data, org_width, org_height = load_image(img_path, WIDTH, HEIGHT)

    # compute time
    for i in range(5):
        start = int(round(time.time() * 1000))
        # Inference
        # encoder
        enc_input_blobs = enc_net.get_input_blob_list()
        enc_net.set_input_blob_data(input_data, enc_input_blobs[0])
        enc_net.update()
        features = enc_net.get_results()

        # decoder
        dec_inputs_blobs = dec_net.get_input_blob_list()
        for f_idx in range(len(features)):
            dec_net.set_input_blob_data(
                features[f_idx], dec_inputs_blobs[f_idx]
            )
        dec_net.update()
        preds_ailia = dec_net.get_results()
        end = int(round(time.time() * 1000))
        print(f"ailia processing time {end-start} ms")

    disp = preds_ailia[-1]
    result_plot(disp, org_width, org_height)
    

def main():
    # Net initialization
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    enc_net = ailia.Net(ENC_MODEL_PATH, ENC_WEIGHT_PATH, env_id=env_id)
    dec_net = ailia.Net(DEC_MODEL_PATH, DEC_WEIGHT_PATH, env_id=env_id)

    recognize_from_image(enc_net, dec_net, IMG_PATH)

    print('Successfully completed!')
    
    
if __name__ == "__main__":
    main()

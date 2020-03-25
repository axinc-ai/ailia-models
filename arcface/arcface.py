import time
import os
import urllib.request

import cv2
import numpy as np

import ailia


# correct pair
IMG_PATH_1 = 'correct_pair_1.jpg'
IMG_PATH_2 = 'correct_pair_2.jpg'
# IMG_PATH_2 = 'incorrect.jpg'

# the threshold was calculated by the `test_performance` function in `test.py`
# of the original repository
THRESHOLD = 0.25572845  


WEIGHT_PATH = 'arcface.onnx'
MODEL_PATH = 'arcface.onnx.prototxt'

RMT_CKPT = "https://storage.googleapis.com/ailia-models/arcface/"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(RMT_CKPT + MODEL_PATH, MODEL_PATH)
if not os.path.exists(WEIGHT_PATH):
    urllib.request.urlretrieve(RMT_CKPT + WEIGHT_PATH, WEIGHT_PATH)


def load_image(img_path):
    # (ref: https://github.com/ronghuaiyang/arcface-pytorch/issues/14)
    # use origin image and fliped image to infer,
    # and concat the feature as the final feature of the origin image.
    image = cv2.imread(img_path, 0)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image = image / 127.5 - 1.0
    return image


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    
def main():
    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    imgs_1 = load_image(IMG_PATH_1)
    imgs_2 = load_image(IMG_PATH_2)
    imgs = np.concatenate([imgs_1, imgs_2], axis=0)
    
    # compute time
    for i in range(1):
        start = int(round(time.time() * 1000))
        preds_ailia = net.predict(imgs)
        end = int(round(time.time() * 1000))
        print("ailia processing time {} ms".format(end - start))

    fe_1 = np.concatenate([preds_ailia[0], preds_ailia[1]], axis=0)
    fe_2 = np.concatenate([preds_ailia[2], preds_ailia[3]], axis=0)
    sim = cosin_metric(fe_1, fe_2)

    print('Similarity of (' + IMG_PATH_1 + ', ' + IMG_PATH_2 + f') : {sim}')
    if THRESHOLD > sim:
        print('They are not the same face!')
    else:
        print('They are the same face!')
    

if __name__ == "__main__":
    main()

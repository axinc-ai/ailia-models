import time
import os
import urllib.request

import cv2

import ailia
from utils import *


def main():
    img_path = 'input.png'
        
    weight_path = 'blazeface.onnx'
    model_path = 'blazeface.onnx.prototxt'

    rmt_ckpt = "https://storage.googleapis.com/ailia-models/blazeface/"

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
    if not os.path.exists(weight_path):
        urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id (0: cpu, 1: gpu): {env_id}')
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    # prepare input data    
    org_img = cv2.imread(img_path)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    imgs = load_image(img_path)

    # compute time
    for i in range(5):
        start = int(round(time.time() * 1000))
        # preds_ailia = net.predict(imgs)
        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(imgs, input_blobs[0])
        net.update()
        preds_ailia = net.get_results()
        end = int(round(time.time() * 1000))
        print("ailia processing time {} ms".format(end - start))

    # Postprocess
    detections = postprocess(preds_ailia)

    # generate detections
    for detection in detections:
        plot_detections(org_img, detection)

    
if __name__ == "__main__":
    main()


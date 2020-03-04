import time
import os
import urllib.request

import cv2
import numpy as np

import ailia


def main():
    weight_path = 'human-pose-estimation-3d.onnx'
    model_path = 'human-pose-estimation-3d.onnx.prototxt'

    rmt_ckpt = "https://storage.googleapis.com/ailia-models/" +\
        "lightweight-human-pose-estimation-3d/"

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
    if not os.path.exists(weight_path):
        urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id (0: cpu, 1: gpu): {env_id}')
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    # FIXME
    imgs = np.random.rand(1, 3, 256, 448)
    
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

    print(len(preds_ailia))
    print(preds_ailia[0].shape)
    print(preds_ailia[1].shape)
    print(preds_ailia[2].shape)


if __name__ == "__main__":
    main()

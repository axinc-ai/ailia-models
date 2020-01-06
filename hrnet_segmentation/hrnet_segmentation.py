import time
import os
import argparse
import urllib.request

from PIL import Image
import numpy as np
import cv2

from utils import save_pred
import ailia


model_names = ['HRNetV2-W48', 'HRNetV2-W18-Small-v1', 'HRNetV2-W18-Small-v2']
img_path = "test.png"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--arch', '-a', metavar="ARCH", default='HRNetV2-W18-Small-v2',
    choices=model_names,
    help='model architecture:  ' + ' | '.join(model_names) + ' (default: HRNetV2-W18-Small-v2)'
)
parser.add_argument(
    '--smooth', '-s', action='store_true',
    help='result image will be smoother by applying bilinear upsampling'
)
args = parser.parse_args()
model_name = args.arch


weight_path = model_name + ".onnx"
model_path = weight_path + ".prototxt"

rmt_ckpt = "https://storage.googleapis.com/ailia-models/hrnet/"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)

print("Weight path: " + weight_path)


# load dataset
img = Image.open(img_path).resize((1024, 512))
img = np.array(img)
img = np.array([img.transpose(2, 0, 1) / 255])

# net initialize
env_id = ailia.get_gpu_environment_id()
print("Environment mode: {} (-1: CPU, 1: GPU)".format(env_id))

net = ailia.Net(model_path, weight_path, env_id=env_id)

# compute time
for i in range(10):
    start = int(round(time.time() * 1000))
    input_blobs = net.get_input_blob_list()
    net.set_input_blob_data(img, input_blobs[0])
    net.update()
    result = net.get_results()
    result = np.asarray(result).reshape(1, 19, 128, 256)

    # if you want to get upsampled image
    if args.smooth:
        result_ = np.zeros((1, 19, 512, 1024))
        for i in range(19):
            result_[0, i] = cv2.resize(
                result[0, i],
                (1024, 512),
                interpolation=cv2.INTER_LINEAR
            )
        result = result_
    
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end-start))

# prediction saving
save_pred(result, ".", ["result"])
print('Successfully finished !')

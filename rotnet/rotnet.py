import os
import argparse
import urllib
import time

import cv2
import numpy as np


import ailia
from utils import generate_rotated_image, visualize


img_path = 'test.jpg'
width, height = 224, 224

model_names = ['mnist', 'gsv2']
model_dict = {
    'mnist': "rotnet_mnist",
    'gsv2': "rotnet_gsv_2"
}


# argument
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m', metavar='MODEL', default='gsv2', choices=model_names,
    help='choose model : ' + ' | '.join(model_names) + ' (default gsv2)'
)
args = parser.parse_args()


model_name = model_dict[args.model]
weight_path = model_name + '.onnx'
model_path = weight_path + '.prototxt'

# downloading
rmt_ckpt = 'https://storage.googleapis.com/ailia-models/rotnet/'
if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)

    
def main():
    # create model
    env_id = ailia.get_gpu_environment_id()
    print("Environment mode: {} (-1: CPU, 1: GPU)".format(env_id))
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    # image processing
    org_img = cv2.imread(img_path)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    
    # generating input image (rotated image)
    rotation_angle = np.random.randint(360)
    rotated_image = generate_rotated_image(
        org_img,
        rotation_angle,
        size=(height, width),
        crop_center=True,
        crop_largest_rect=True
    )

    input_data = rotated_image.reshape((1, 224, 224, 3))
    net.set_input_shape(input_data.shape)
    
    # compute time
    for i in range(10):
        start = int(round(time.time() * 1000))
        pred = net.predict(input_data)
        end = int(round(time.time() * 1000))
        print(f"ailia processing time {end-start} ms")
    predicted_angle = np.argmax(pred, axis=1)[0]
    print(f"predicted angle is {predicted_angle}")

    # visualize result
    visualize(rotated_image, rotation_angle, predicted_angle, height, width)
    print('Successfully finished!')


if __name__ == "__main__":
    main()

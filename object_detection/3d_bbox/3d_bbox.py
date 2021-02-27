#ailia predict api sample

import numpy as np
import time
import os
import sys
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import argparse

import ailia
import ailia_classifier_labels

from lib_3d_bbox.Dataset import *
from lib_3d_bbox.Plotting import *
from lib_3d_bbox import ClassAverages
from lib_3d_bbox.Math import *

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import plot_results, write_predictions, load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# settings
weight_path_yolov3 = "yolov3.opt.onnx"
model_path_yolov3 = "yolov3.opt.onnx.prototxt"

weight_path_3d_bbox = "3d_bbox.onnx"
model_path_3d_bbox = "3d_bbox.onnx.prototxt"

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

calib_file = "./lib_3d_bbox/calib_cam_to_cam.txt"

COCO_CATEGORY = [
    "Pedestrian", "bicycle", "Car", "motorcycle", "airplane", "bus", "train",
    "Truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
THRESHOLD = 0.4
IOU = 0.45

# Default input size
height = 224
width = 224
channels = 3
batch_size = 1

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")

FLAGS = parser.parse_args()
averages = ClassAverages.ClassAverages()
angle_bins = generate_bins(2)

# ======================
# Main functions
# ======================
def recognize_from_image():

env_id = ailia.get_gpu_environment_id()
net_yolov3 = ailia.Detector(
        model_path_yolov3,
        weight_path_yolov3,
        len(COCO_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=env_id,
    )
net_3d_bbox = ailia.Net(model_path_3d_bbox,weight_path_3d_bbox,env_id=env_id)
net_yolov3.set_input_shape(width, height)
start_time = time.time()

truth_img = cv2.imread(file_name)
img = np.copy(truth_img)
yolo_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

detections = net_yolov3.run(yolo_img, THRESHOLD, IOU)

for detection in detections:
    print(detection)
    detect_class = COCO_CATEGORY[detection[0]]
    xmin = int(detection[2][0] * img.shape[1])
    ymin = int(detection[2][1] * img.shape[0])
    xmax = int(detection[2][2] * img.shape[1]) + int(detection[2][0] * img.shape[1])
    ymax = int(detection[2][3] * img.shape[0]) + int(detection[2][1] * img.shape[0])
    detect_box_2d = [(xmin, ymin),(xmax, ymax)]
    if not averages.recognized_class(detect_class):
        continue

    # this is throwing when the 2d bbox is invalid
    # TODO: better check
    try:
        detectedObject = DetectedObject(img, detect_class, detect_box_2d, calib_file)
    except:
        continue

    theta_ray = detectedObject.theta_ray
    input_img = detectedObject.img
    proj_matrix = detectedObject.proj_matrix
    box_2d = detect_box_2d
    detected_class = detect_class

    input_tensor = np.expand_dims(input_img,0)

    [orient, conf, dim] = net_3d_bbox.run(input_tensor)
    orient = orient[0, :, :]
    conf = conf[0, :]
    dim = dim[0, :]
    dim += averages.get_item(detected_class)

    argmax = np.argmax(conf)
    orient = orient[argmax, :]
    cos = orient[0]
    sin = orient[1]
    alpha = np.arctan2(sin, cos)
    alpha += angle_bins[argmax]
    alpha -= np.pi

    if FLAGS.show_yolo:
        location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
    else:
        location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

    if not FLAGS.hide_debug:
        print('Estimated pose: %s'%location)

if FLAGS.show_yolo:
    numpy_vertical = np.concatenate((truth_img, img), axis=0)
    cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
else:
    cv2.imshow('3D detections', img)

if FLAGS.video:
    cv2.waitKey(1)
else:
    if cv2.waitKey(0) != 32:  # space bar
        exit()





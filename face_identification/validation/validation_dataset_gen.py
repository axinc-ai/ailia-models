import sys
import time
import argparse
import os

import cv2

import ailia

# import original modules
sys.path.append('../../util')
from utils import check_file_existance  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

sys.path.append('../../face_detection/blazeface')
from blazeface_utils import *

# ======================
# Parameters
# ======================
MODEL_LISTS = ["yolov3","yolov3-masked-face","blazeface"]

YOLOV3_THRESHOLD = 0.1
YOLOV3_IOU = 0.45

BLAZEFACE_INPUT_IMAGE_HEIGHT = 128
BLAZEFACE_INPUT_IMAGE_WIDTH = 128

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Face cropping for arcface validation'
)
parser.add_argument(
    '-i', '--input', metavar='INPUT FOLDER',
    default=None,
    help='The input folder path. '
)
parser.add_argument(
    '-o', '--output', metavar='OUTPUT FOLDER',
    default=None,
    help='The output folder path. '
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='blazeface', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
args = parser.parse_args()

if args.arch == "blazeface":
    REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
    WEIGHT_PATH = 'blazeface.onnx'
    MODEL_PATH = 'blazeface.onnx.prototxt'
    FACE_CATEGORY = ['face']
elif args.arch == "yolov3":
    REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3-face/'
    WEIGHT_PATH = 'yolov3-face.opt.onnx'
    MODEL_PATH = 'yolov3-face.opt.onnx.prototxt'
    FACE_CATEGORY = ['face']
elif args.arch == "yolov3-masked-face":
    REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/yolov3-face/'
    WEIGHT_PATH = 'ax_masked_face.opt.onnx'
    MODEL_PATH = 'ax_masked_face.opt.onnx.prototxt'
    FACE_CATEGORY = ['masked','half','no_mask']

# ======================
# Main functions
# ======================

def recognize_from_image(detector,dst_path,src_dir,file_):
    # prepare input data
    #img = load_image(src_dir+"/"+file_)

    img = cv2.imread(src_dir+"/"+file_)
    h, w = img.shape[0], img.shape[1]

    if args.arch=="yolov3":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        detector.compute(img, YOLOV3_THRESHOLD, YOLOV3_IOU)
        count = detector.get_object_count()
    else:
        # prepare input data
        img = cv2.imread(src_dir+"/"+file_)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (BLAZEFACE_INPUT_IMAGE_WIDTH, BLAZEFACE_INPUT_IMAGE_HEIGHT))
        image = image.transpose((2, 0, 1))  # channel first
        image = image[np.newaxis, :, :, :]  # (batch_size, channel, h, w)
        input_data = image / 127.5 - 1.0

        # inference
        preds_ailia = detector.predict([input_data])

        # postprocessing
        detections = postprocess(preds_ailia)
        count = len(detections)
    
    texts = []
    written = False
    for idx in range(count):
        if args.arch=="yolov3":
            # get detected face
            obj = detector.get_object(idx)
            margin = 1.0
        else:
            # get detected face
            obj = detections[idx]
            d = obj[0]
            obj = ailia.DetectorObject(
                category = 0,
                prob = 1.0,
                x = d[1],
                y = d[0],
                w = d[3]-d[1],
                h = d[2]-d[0] )
            margin = 1.4
    
        cx = (obj.x + obj.w/2) * w
        cy = (obj.y + obj.h/2) * h
        cw = max(obj.w * w * margin,obj.h * h * margin)
        fx = max(cx - cw/2, 0)
        fy = max(cy - cw/2, 0)
        fw = min(cw, w-fx)
        fh = min(cw, h-fy)
        top_left = (int(fx), int(fy))
        bottom_right = (int((fx+fw)), int(fy+fh))

        print("face detected "+str(top_left)+"-"+str(bottom_right))

        # get detected face
        crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], 0:3]
        if crop_img.shape[0]<=0 or crop_img.shape[1]<=0:
            continue
        cv2.imwrite(dst_path, crop_img)
        written = True
    
    if not written:
        print("face not found")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # check folder existing
    if not os.path.exists(args.input):
        print("error : directory not found "+args.input)
        sys.exit(1)
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    if args.arch == 'blazeface':
        detector = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        detector = ailia.Detector(
            MODEL_PATH,
            WEIGHT_PATH,
            len(FACE_CATEGORY),
            format=ailia.NETWORK_IMAGE_FORMAT_RGB,
            channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
            range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
            algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
            env_id=env_id
        )

    # process images
    no = 0
    for src_dir, dirs, files in os.walk(args.input):
        files = sorted(files)
        for file_ in files:
            root, ext = os.path.splitext(file_)

            if file_==".DS_Store":
                continue
            if file_=="Thumbs.db":
                continue
            if not(ext == ".jpg" or ext == ".png" or ext == ".bmp"):
                continue

            print(src_dir+"/"+file_)
            folders=src_dir.split("/")
            folder=folders[len(folders)-1]
            dst_dir = args.output+"/"+folder
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            dst_path= dst_dir+ "/"+str(no)+".jpg"
            recognize_from_image(detector,dst_path,src_dir,file_)
            no=no+1

if __name__ == '__main__':
    main()

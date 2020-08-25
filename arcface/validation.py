import sys
import time
import argparse
import os
import re

import numpy as np
import cv2

import ailia

# import original modules
sys.path.append('../util')
from webcamera_utils import adjust_frame_size  # noqa: E402
from image_utils import load_image, draw_result_on_img  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import check_file_existance  # noqa: E402
from detector_utils import hsv_to_rgb # noqa: E402C

import matplotlib.pyplot as plt

# ======================
# PARAMETERS
# ======================

MODEL_LISTS = ['arcface', 'arcface_mixed10', 'arcface_mixed150', 'arcface_mixed_90_82', 'arcface_mixed_90_99', 'arcface_mixed_eq_90_89']

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/arcface/"

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# (IMAGE_HEIGHT * 2 * WEBCAM_SCALE, IMAGE_WIDTH * 2 * WEBCAM_SCALE)
# Scale to determine the input size of the webcam
WEBCAM_SCALE = 1.5

# the threshold was calculated by the `test_performance` function in `test.py`
# of the original repository
THRESHOLD = 0.25572845

# ======================
# Arguemnt Parser Config
# ======================
parser = argparse.ArgumentParser(
    description='Determine if the person is the same from two facial images.'
)
parser.add_argument(
    '-i', '--input', metavar='INPUT FOLDER',
    default=None,
    help='The input folder path. ' +
         'Create confusion matrix.'
)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='arcface', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-s', '--skip',
    action='store_true',
    help='RCalculate using only some images'
)
args = parser.parse_args()

WEIGHT_PATH = args.arch+'.onnx'
MODEL_PATH = args.arch+'.onnx.prototxt'

# ======================
# Utils
# ======================
def preprocess_image(image, input_is_bgr=False):
    # (ref: https://github.com/ronghuaiyang/arcface-pytorch/issues/14)
    # use origin image and fliped image to infer,
    # and concat the feature as the final feature of the origin image.
    if input_is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if "eq_" in args.arch:
        image = cv2.equalizeHist(image)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    return image / 127.5 - 1.0  # normalize


def prepare_input_data(image_path):
    image = load_image(
        image_path,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        rgb=False,
        normalize_type='None'
    )
    return preprocess_image(image)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def get_evaluation_files(input):
    before_folder=""
    folder_cnt=0

    file_dict={}
    file_list=[]

    for src_dir, dirs, files in os.walk(input):
        #files = sorted(files)
        files = sorted(files,key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        for file_ in files:
            root, ext = os.path.splitext(file_)

            if file_==".DS_Store":
                continue
            if file_=="Thumbs.db":
                continue
            if not(ext == ".jpg" or ext == ".png" or ext == ".bmp"):
                continue

            folders=src_dir.split("/")
            folder=folders[len(folders)-1]
            before_folder=folder
            if not(folder in file_dict):
                file_dict[folder]=[]
                folder_cnt=folder_cnt+1
            if args.skip:
                NUM_SKIP_PER_PERSON = 4
                if(len(file_dict[folder])>=NUM_SKIP_PER_PERSON):
                    continue
                NUM_SKIP_PERSON = 16
                if folder_cnt >= NUM_SKIP_PERSON:
                    continue
            file_dict[folder].append(src_dir+"/"+file_)
            file_list.append(src_dir+"/"+file_)
    
    return file_list


def get_feature_values(net,file_list):
    BATCH_SIZE = net.get_input_shape()[0]
    fe_list=[]
    for i in range(0,len(file_list)):
        inputs0=file_list[i]
        print(inputs0)
        imgs_1 = prepare_input_data(inputs0)
        if BATCH_SIZE==4:
           imgs_1 = np.concatenate([imgs_1, imgs_1], axis=0)
        preds_ailia1 = net.predict(imgs_1)
        fe_1 = np.concatenate([preds_ailia1[0], preds_ailia1[1]], axis=0)
        fe_list.append(fe_1)
    return fe_list


def display_result(file_list,fe_list):
    fig = plt.figure(figsize=(12.0, 12.0))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)

    ax1.tick_params(labelbottom="on")
    ax2.tick_params(labelleft="on")
    ax3.tick_params(labelleft="on")

    max_cnt=len(file_list)

    x=np.zeros((max_cnt))
    y=np.zeros((max_cnt))
    t=np.zeros((max_cnt))

    heatmap=np.zeros((len(file_list),len(file_list)))
    expect=np.zeros((len(file_list),len(file_list)))
    detected=np.zeros((len(file_list),len(file_list)))

    success = 0
    failed = 0

    for i0 in range(0,len(file_list)):
        for i1 in range(0,len(file_list)):
            inputs0=file_list[i0]
            inputs1=file_list[i1]

            # postprocessing
            fe_1 = fe_list[i0]
            fe_2 = fe_list[i1]
            sim = cosin_metric(fe_1, fe_2)

            ex=0
            f0=inputs0.split("/")
            f1=inputs1.split("/")

            f0=f0[len(f0)-2]
            f1=f1[len(f1)-2]

            if f0==f1:
                ex=1
            
            print(f'Similarity of ({inputs0}, {inputs1}) : {sim:.3f}')
            if THRESHOLD > sim:
                print('They are not the same face!')
            else:
                print('They are the same face!')
            
            if (f0==f1 and THRESHOLD <= sim) or (f0!=f1 and THRESHOLD > sim):
                success = success + 1
            else:
                failed = failed + 1

            heatmap[int(i0),int(i1)]=sim
            expect[int(i0),int(i1)]=ex
            if THRESHOLD <= sim:
                detected[int(i0),int(i1)]=1
            else:
                detected[int(i0),int(i1)]=0
    
    accuracy = int(success * 10000 / (success + failed))/100

    ax1.pcolor(expect, cmap=plt.cm.Blues)
    ax2.pcolor(detected, cmap=plt.cm.Blues)
    ax3.pcolor(heatmap, cmap=plt.cm.Blues)

    if False:   # Plot values
        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                if heatmap[y, x]!=0:
                    ax2.text(x + 0.5, y + 0.5, str(heatmap[y, x]),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8
                    )

    ax1.set_title('expected ')
    ax1.set_xlabel('(face2)')
    ax1.set_ylabel('(face1)')
    ax1.legend(loc='upper right')

    ax2.set_title('detected (threshold '+str(THRESHOLD)+' accuracy '+str(accuracy)+' %)')
    ax2.set_xlabel('(face2)')
    ax2.set_ylabel('(face1')
    ax2.legend(loc='upper right')

    ax3.set_title('similality')
    ax3.set_xlabel('(face2)')
    ax3.set_ylabel('(face1')
    ax3.legend(loc='upper right')

    fig.savefig("confusion_"+args.arch+".png",dpi=100)


# ======================
# Main functions
# ======================

def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # check folder
    if args.input==None or not os.path.exists(args.input):
        print("Input folder not found")
        return

    # get target files
    file_list = get_evaluation_files(args.input)

    # net initialize
    env_id = ailia.get_gpu_environment_id()
    print(f'env_id: {env_id}')
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # get feature values
    fe_list = get_feature_values(net,file_list)

    # create confusion matrix
    display_result(file_list,fe_list)

if __name__ == "__main__":
    main()

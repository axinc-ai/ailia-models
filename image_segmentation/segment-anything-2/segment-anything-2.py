import sys
import os
import time
from logging import getLogger

import numpy as np
import cv2

import ailia

import os
import numpy as np
import matplotlib.pyplot as plt
import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_IMAGE_ENCODER_L_PATH = 'image_encoder_hiera_l.onnx'
MODEL_IMAGE_ENCODER_L_PATH = 'image_encoder_hiera_l.onnx.prototxt'
WEIGHT_PROMPT_ENCODER_L_PATH = 'prompt_encoder_hiera_l.onnx'
MODEL_PROMPT_ENCODER_L_PATH = 'prompt_encoder_hiera_l.onnx.prototxt'
WEIGHT_MASK_DECODER_L_PATH = 'mask_decoder_hiera_l.onnx'
MODEL_MASK_DECODER_L_PATH = 'mask_decoder_hiera_l.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/segment-anything-2/'

IMAGE_PATH = 'truck.jpg'
SAVE_IMAGE_PATH = 'output.png'

POINT1 = (500, 375)
POINT2 = (1125, 625)

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Segment Anything 2', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-p', '--pos', action='append', type=int, metavar="X", nargs=2,
    help='Positive coordinate specified by x,y.'
)
parser.add_argument(
    '--neg', action='append', type=int, metavar="X", nargs=2,
    help='Negative coordinate specified by x,y.'
)
parser.add_argument(
    '--box', type=int, metavar="X", nargs=4,
    help='Box coordinate specified by x1,y1,x2,y2.'
)
parser.add_argument(
    '--idx', type=int, choices=(0, 1, 2, 3),
    help='Select mask index.'
)
parser.add_argument(
    '-m', '--model_type', default='hiera_l', choices=('hiera_l'),
    help='Select model.'
)
parser.add_argument(
    '--onnx', action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

# ======================
# Utility
# ======================

#np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, savepath = None):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if i == 0:
            plt.savefig(savepath)

# ======================
# Logic
# ======================

from sam2_image_predictor import SAM2ImagePredictor
#from sam2_video_predictor import SAM2VideoPredictor

# ======================
# Main
# ======================

def recognize_from_image(image_encoder, prompt_encoder, mask_decoder):
    pos_points = args.pos
    neg_points = args.neg
    box = args.box

    if pos_points is None:
        if neg_points is None and box is None:
            pos_points = [POINT1]
        else:
            pos_points = []
    if neg_points is None:
        neg_points = []
    #if box is not None:
    #    box = np.array(box).reshape(2, 2)

    input_point = []
    input_label = []
    if pos_points:
        input_point.append(np.array(pos_points))
        input_label.append(np.ones(len(pos_points)))
    if neg_points:
        input_point.append(np.array(neg_points))
        input_label.append(np.zeros(len(neg_points)))
    input_point = np.array(input_point)
    input_label = np.array(input_label)

    image_predictor = SAM2ImagePredictor()


    for image_path in args.input:
        image = cv2.imread(image_path)
        orig_hw = [image.shape[0], image.shape[1]]


        features = image_predictor.set_image(image, image_encoder, args.onnx)

        masks, scores, logits = image_predictor.predict(
            orig_hw=orig_hw,
            features=features,
            point_coords=input_point,
            point_labels=input_label,
            box=box,
            multimask_output=True,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            onnx=args.onnx
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True, savepath=savepath)


def recognize_from_video(image_encoder, prompt_encoder, mask_decoder):
    #video_predictor = SAM2VideoPredictor()
    raise("not implemented yet")

def main():
    # model files check and download
    check_and_download_models(WEIGHT_IMAGE_ENCODER_L_PATH, None, REMOTE_PATH)
    check_and_download_models(WEIGHT_PROMPT_ENCODER_L_PATH, None, REMOTE_PATH)
    check_and_download_models(WEIGHT_MASK_DECODER_L_PATH, None, REMOTE_PATH)

    if args.onnx:
        import onnxruntime
        image_encoder = onnxruntime.InferenceSession(WEIGHT_IMAGE_ENCODER_L_PATH)
        prompt_encoder = onnxruntime.InferenceSession(WEIGHT_PROMPT_ENCODER_L_PATH)
        mask_decoder = onnxruntime.InferenceSession(WEIGHT_MASK_DECODER_L_PATH)
    else:
        image_encoder = ailia.Net(weight=WEIGHT_IMAGE_ENCODER_L_PATH, stream=None, memory_mode=11, env_id=1)
        prompt_encoder = ailia.Net(weight=WEIGHT_PROMPT_ENCODER_L_PATH, stream=None, memory_mode=11, env_id=1)
        mask_decoder = ailia.Net(weight=WEIGHT_MASK_DECODER_L_PATH, stream=None, memory_mode=11, env_id=1)

    if args.video is not None:
        recognize_from_video(image_encoder, prompt_encoder, mask_decoder)
    else:
        recognize_from_image(image_encoder, prompt_encoder, mask_decoder)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()

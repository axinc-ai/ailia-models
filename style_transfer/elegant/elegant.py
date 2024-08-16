import os
import cv2
import sys
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import ailia

# Import original modules

from elegant_util import Inference
sys.path.append('../../style_transfer/psgan')
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # NOQA: E402
from image_utils import imread  # NOQA: E402
from webcamera_utils import get_writer,adjust_frame_size, get_capture, cut_max_square  # NOQA: E402


# Logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
ELEGANT_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/elegant/"
ELEGANT1_WEIGHT_PATH = "elegant1.onnx"
ELEGANT1_MODEL_PATH = "elegant1.onnx.prototxt"

ELEGANT2_WEIGHT_PATH = "elegant2.onnx"
ELEGANT2_MODEL_PATH = "elegant2.onnx.prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/psgan/"
FACE_PARSER_WEIGHT_PATH = "face_parser.onnx"
FACE_PARSER_MODEL_PATH = "face_parser.onnx.prototxt"
face_parser_path = [FACE_PARSER_MODEL_PATH, FACE_PARSER_WEIGHT_PATH] 

BLAZEFACE_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/blazeface/"
FACE_DETECTOR_WEIGHT_PATH = "blazeface.onnx"
FACE_DETECTOR_MODEL_PATH = "blazeface.onnx.prototxt"
detector_parser_path = [FACE_DETECTOR_MODEL_PATH, FACE_DETECTOR_WEIGHT_PATH] 


FACE_ALIGNMENT_REMOTE_PATH = "https://storage.googleapis.com/ailia-models/face_alignment/"
FACE_ALIGNMENT_WEIGHT_PATH = "2DFAN-4.onnx"
FACE_ALIGNMENT_MODEL_PATH = "2DFAN-4.onnx.prototxt"
face_aligment_path = [FACE_ALIGNMENT_MODEL_PATH, FACE_ALIGNMENT_WEIGHT_PATH] 

SOURCE_IMAGE_PATH = "input.png"
REFERENCE_IMAGE_PATH = "reference.png"
SAVE_IMAGE_PATH = "output.png"
IMAGE_HEIGHT = 361
IMAGE_WIDTH = 361

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    "EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer",
    SOURCE_IMAGE_PATH,
    SAVE_IMAGE_PATH,
)

 
parser.add_argument("-r","--reference", nargs='*', default=["reference.png"])
parser.add_argument(
    "--onnx",
    action="store_true",
    help="Execute Onnx Runtime mode.",
)
parser.add_argument(
    "--use_dlib",
    action="store_true",
    help="Use dlib models for inference.",
)

args = update_parser(parser)

# ======================
# Main functions
# ======================

def compute(args,inference,imgA,imgB):
    #from config import get_config
   
    imgA_RGB = cv2.cvtColor(imgA,cv2.COLOR_BGR2RGB)
    imgB_RGB = cv2.cvtColor(imgB,cv2.COLOR_BGR2RGB)

    result = inference.transfer(imgA_RGB, imgB_RGB, postprocess=True) 
    h, w, _ = imgA.shape
    result = cv2.resize(result,(h,w))
    #vis_image = np.hstack((imgA, imgB, result))
    vis_image = result
    return vis_image

def transfer_to_image():
    # Prepare input data
    # Net initialize
    inference = Inference(args,face_parser_path,detector_parser_path,face_aligment_path)

    # Inference
    for i, (imga_name, imgb_name) in enumerate(zip(args.input, args.reference)):
        logger.info("Start inference...")

        imgA = imread(imga_name)
        imgB = imread(imgb_name)
        if args.benchmark:
            logger.info("BENCHMARK mode")
            for i in range(5):
                start = int(round(time.time() * 1000))
                vis_image = compute(args,inference,imgA,imgB)
                end = int(round(time.time() * 1000))
                logger.info(f"\tailia processing time {end - start} ms")
        else:
            vis_image = compute(args,inference,imgA,imgB)

        savepath = get_savepath(args.savepath, args.input[0])

        vis_image = vis_image.astype(np.uint8)
        vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepath,vis_image.astype(np.uint8))
        # Postprocessing
        logger.info(f"saved at : {savepath}")
 
    logger.info("Script finished successfully.")


def transfer_to_video():
    # Net initialize

    inference = Inference(args,face_parser_path,detector_parser_path,face_aligment_path)
    capture = get_capture(args.video)
    imgB = cv2.imread(args.reference[0])
    
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()

        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        # Inference
        vis_image = compute(args,inference,frame,imgB)

        # Postprocessing
        vis_image = vis_image.astype(np.uint8)
        vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)
        frame_shown = True

        cv2.imshow("frame", vis_image)

        # save results
        if writer is not None:
            writer.write(vis_image)

    capture.release()
    cv2.destroyAllWindows()
    logger.info("Script finished successfully.")


def main():
    # Check model files and download
    check_and_download_models(
        FACE_PARSER_WEIGHT_PATH,
        FACE_PARSER_MODEL_PATH,
        REMOTE_PATH,
    )
    check_and_download_models(
        FACE_DETECTOR_WEIGHT_PATH,
        FACE_DETECTOR_MODEL_PATH,
        BLAZEFACE_REMOTE_PATH,
    )
    check_and_download_models(
        FACE_ALIGNMENT_WEIGHT_PATH,
        FACE_ALIGNMENT_MODEL_PATH,
        FACE_ALIGNMENT_REMOTE_PATH,
    )
    check_and_download_models(
        ELEGANT1_WEIGHT_PATH,
        ELEGANT1_MODEL_PATH,
        ELEGANT_REMOTE_PATH,
    )
    check_and_download_models(
        ELEGANT2_WEIGHT_PATH,
        ELEGANT2_MODEL_PATH,
        ELEGANT_REMOTE_PATH,
    )

    if args.video is not None:
        # Video mode
        transfer_to_video()
    else:
        # Image mode
        transfer_to_image()


if __name__ == "__main__":
    main()

import sys
import os
import time
from copy import deepcopy
from collections import OrderedDict
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import urlretrieve, progress_print, check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = "edge_sam_3x_encoder.onnx"
WEIGHT_DEC_PATH = "edge_sam_3x_decoder.onnx"
MODEL_ENC_PATH = "edge_sam_3x_encoder.onnx.prototxt"
MODEL_DEC_PATH = "edge_sam_3x_decoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/edge_sam/"

IMAGE_PATH = "truck.jpg"
SAVE_IMAGE_PATH = "output.png"

POINT = (500, 375)

TARGET_LENGTH = 1024

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("Edge SAM", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    "-p",
    "--pos",
    action="append",
    type=int,
    metavar="X",
    nargs=2,
    help="Positive coordinate specified by x,y.",
)
parser.add_argument(
    "--neg",
    action="append",
    type=int,
    metavar="X",
    nargs=2,
    help="Negative coordinate specified by x,y.",
)
parser.add_argument(
    "--box",
    type=int,
    metavar="X",
    nargs=4,
    help="Box coordinate specified by x1,y1,x2,y2.",
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


def apply_coords(coords, h, w):
    scale = TARGET_LENGTH / max(h, w)
    coords = deepcopy(coords).astype(np.float32)
    coords[..., 0] = coords[..., 0] * scale
    coords[..., 1] = coords[..., 1] * scale

    return coords


def show_mask(mask, img):
    color = np.array([255, 144, 30])
    color = color.reshape(1, 1, -1)

    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1)

    mask_image = mask * color
    img = (img * ~mask) + (img * mask) * 0.6 + mask_image * 0.4

    return img


def show_points(coords, labels, img):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    for p in pos_points:
        cv2.drawMarker(
            img,
            p,
            (0, 255, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            line_type=cv2.LINE_AA,
            markerSize=30,
            thickness=5,
        )
    for p in neg_points:
        cv2.drawMarker(
            img,
            p,
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            line_type=cv2.LINE_AA,
            markerSize=30,
            thickness=5,
        )

    return img


def show_box(box, img):
    cv2.rectangle(
        img,
        box[0],
        box[1],
        color=(2, 118, 2),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0,
    )

    return img


# ======================
# Main functions
# ======================


def preprocess(img):
    im_h, im_w, _ = img.shape

    scale = TARGET_LENGTH / max(im_h, im_w)
    ow = int(im_w * scale)
    oh = int(im_h * scale)
    img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BICUBIC))

    img = normalize_image(img, normalize_type="ImageNet")

    pad_h = TARGET_LENGTH - oh
    pad_w = TARGET_LENGTH - ow
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), "constant")

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img, (oh, ow)


def postprocess_masks(
    mask: np.ndarray, input_h: int, input_w: int, orig_h: int, orig_w: int
):
    mask = mask.squeeze(0).transpose(1, 2, 0)
    mask = cv2.resize(
        mask, (TARGET_LENGTH, TARGET_LENGTH), interpolation=cv2.INTER_LINEAR
    )
    mask = mask[:input_h, :input_w, :]
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = mask.transpose(2, 0, 1)[None, :, :, :]
    return mask


def predict(models, img, point_coords, point_labels):
    im_h, im_w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB
    img, input_size = preprocess(img)

    encoder = models["encoder"]
    if not args.onnx:
        output = encoder.predict([img])
    else:
        output = encoder.run(None, {"image": img})
    features = output[0]

    point_coords = np.array(point_coords, dtype=np.float32)[None]
    point_labels = np.array(point_labels, dtype=np.float32)[None]
    point_coords = apply_coords(point_coords, im_h, im_w)

    decoder = models["decoder"]
    if not args.onnx:
        output = decoder.predict([features, point_coords, point_labels])
    else:
        output = decoder.run(
            None,
            {
                "image_embeddings": features,
                "point_coords": point_coords,
                "point_labels": point_labels,
            },
        )
    scores, low_res_masks = output
    masks = postprocess_masks(low_res_masks, *input_size, im_h, im_w)

    mask_threshold = 0.0
    masks = masks > mask_threshold

    masks = masks.squeeze(0)
    scores = scores.squeeze(0)
    return masks, scores


def recognize_from_image(models):
    pos_points = args.pos
    neg_points = args.neg
    box = args.box

    if pos_points is None:
        if neg_points is None and box is None:
            pos_points = [POINT]
        else:
            pos_points = []
    if neg_points is None:
        neg_points = []
    if box is not None:
        box = np.array(box).reshape(2, 2)

    lf = "\n"
    logger.info(f"Positive coordinate: {pos_points}")
    logger.info(f"Negative coordinate: {neg_points}")
    logger.info(f"Box coordinate: {lf if box is not None else ''}{box}")

    if box is not None:
        coord_list = box
        label_list = [2, 3]
    else:
        coord_list = np.array(pos_points + neg_points)
        label_list = np.array([1 for _ in pos_points] + [0 for _ in neg_points])

    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(models, img, coord_list, label_list)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Logging
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            output = predict(models, img, coord_list, label_list)

        masks, scores = output

        if box is not None:
            mask = np.expand_dims(masks[0], axis=0)
            res_img = show_mask(mask, img)
            res_img = show_box(box, res_img)
        else:
            mask = np.expand_dims(masks[scores.argmax()], axis=0)
            res_img = show_mask(mask, img)
            res_img = show_points(coord_list, label_list, res_img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext=".png")
        cv2.imwrite(savepath, res_img)
        logger.info(f"saved at : {savepath}")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_DEC_PATH, MODEL_DEC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id)
        decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        encoder = onnxruntime.InferenceSession(WEIGHT_ENC_PATH, providers=providers)
        decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH, providers=providers)

    models = dict(
        encoder=encoder,
        decoder=decoder,
    )

    if not args.gui:
        recognize_from_image(models)
    else:
        show_gui(args.input[0])


if __name__ == "__main__":
    main()

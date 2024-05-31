import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def load_image(image_filename: str) -> np.ndarray:
    img_pil = Image.open(image_filename)
    img_pil = img_pil.resize((960, 1024))
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_np, img_pil


def preprocess(img: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)

    # det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img
    det_scale = float(new_height) / img.shape[0]
    return det_img, det_scale


def get_model_file_names() -> dict[str, tuple[str]]:
    return {
        "detection": {
            "weight": "detection.onnx",
            "model": "detection.onnx.prototxt",
        },
        "recognition": {
            "weight": "recognition.onnx",
            "model": "recognition.onnx.prototxt",
        },
        "landmark_2d_106": {
            "weight": "landmark_2d_106.onnx",
            "model": "landmark_2d_106.onnx.prototxt",
        },
        "landmark_3d_68": {
            "weight": "landmark_3d_68.onnx",
            "model": "landmark_3d_68.onnx.prototxt",
        },
        "genderage": {
            "weight": "genderage.onnx",
            "model": "genderage.onnx.prototxt",
        },
        "vae_decoder": {
            "weight": "decoder.onnx",
            "model": "decoder.onnx.prototxt",
        },
        # UNetエクスポートできたらコメントアウト
        # "unet": {
        #     "weight": "unet.onnx",
        #     "model": "unet.onnx.prototxt",
        # },
        "text_encoder": {
            "weight": "text_encoder.onnx",
            "model": "text_encoder.onnx.prototxt",
        },
        "text_encoder_2": {
            "weight": "text_encoder2.onnx",
            "model": "text_encoder2.onnx.prototxt",
        },
        "controlnet": {
            "weight": "controlnet.onnx",
            "model": "controlnet.onnx.prototxt",
        },
        "image_proj_model": {
            "weight": "image_proj_model.onnx",
            "model": "image_proj_model.onnx.prototxt",
        },
    }


def draw_kps(
    image_pil,
    kps,
    color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)],
):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

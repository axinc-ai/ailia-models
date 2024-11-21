import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia


# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from image_utils import normalize_image
from nms_utils import nms_boxes  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

from util_math import *
from util_affine import *

logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_PATH = "generator.onnx"
MODEL_PATH = "generator.onnx.prototxt"
WEIGHT_YOLOV5FACE_PATH = "YoloV5Face.onnx"
MODEL_YOLOV5FACE_PATH = "YoloV5Face.onnx.prototxt"
WEIGHT_FACEMESH_PATH = "FaceMesh.onnx"
MODEL_FACEMESH_PATH = "FaceMesh.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/deepfacelive/"

IMAGE_PATH = "Obama.jpg"
SOURCE_PATH = "Kim Chen Yin.png"
SAVE_IMAGE_PATH = "output.png"

IMG_SIZE = 256


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("DeepFaceLive", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("-src", "--source", default=SOURCE_PATH, help="source image")
parser.add_argument(
    "--window_size",
    type=int,
    default=448,
    help="Image size when detecting. Multiple of 32. 0 is auto.",
)
parser.add_argument(
    "--threshold", type=float, default=0.5, help="face detection threshold. 0.0-1.0"
)
parser.add_argument("--max_faces", type=int, default=1, help="max faces")
parser.add_argument(
    "--temporal_smoothing", type=int, default=1, help="temporal smoothing"
)
parser.add_argument(
    "--marker_coverage", type=float, default=1.4, help="marker coverage"
)
parser.add_argument(
    "--marker_temporal_smoothing", type=int, default=1, help="marker temporal smoothing"
)
parser.add_argument("--face_coverage", type=float, default=2.2, help="face coverage")
parser.add_argument("--resolution", type=int, default=224, help="output resolution")
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Class Definitions
# ======================


class ELandmarks2D(IntEnum):
    L5 = 0
    L68 = 1
    L106 = 2
    L468 = 3


@dataclass
class FLandmarks2D:
    """
    Describes 2D face landmarks in uniform float coordinates
    """

    type: ELandmarks2D = None
    ulmrks: np.ndarray = None


class AlignMode(IntEnum):
    FROM_RECT = 0
    FROM_POINTS = 1
    FROM_STATIC_RECT = 2


@dataclass
class FaceSwapInfo:
    face_urect: np.ndarray = None
    face_pose: np.ndarray = None
    face_ulmrks: FLandmarks2D = None

    face_resolution: int = None

    face_align_image: np.ndarray = None
    face_align_lmrks_mask: np.ndarray = None
    # face_align_mask_name: str = None
    # face_anim_image_name: str = None
    face_swap_image: np.ndarray = None
    # face_swap_mask_name: str = None

    image_to_align_uni_mat: np.ndarray = None
    face_align_ulmrks: FLandmarks2D = None


# ======================
# Secondaty Functions
# ======================


def setup_yolov5face(net):
    def fit_in(
        img,
        TW=None,
        TH=None,
        pad_to_target: bool = False,
        allow_upscale: bool = False,
        interpolation: "ImageProcessor.Interpolation" = None,
    ) -> float:
        """
        fit image in w,h keeping aspect ratio

            TW,TH           int/None     target width,height

            pad_to_target   bool    pad remain area with zeros

            allow_upscale   bool    if image smaller than TW,TH it will be upscaled

            interpolation   ImageProcessor.Interpolation. value

        returns scale float value
        """
        N, H, W, C = img.shape

        if TW is not None and TH is None:
            scale = TW / W
        elif TW is None and TH is not None:
            scale = TH / H
        elif TW is not None and TH is not None:
            SW = W / TW
            SH = H / TH
            scale = 1.0
            if SW > 1.0 or SH > 1.0 or (SW < 1.0 and SH < 1.0):
                scale /= max(SW, SH)
        else:
            raise ValueError("TW or TH should be specified")

        if not allow_upscale and scale > 1.0:
            scale = 1.0

        if scale != 1.0:
            img = img.transpose((1, 2, 0, 3)).reshape((H, W, N * C))
            img = cv2.resize(
                img,
                (int(W * scale), int(H * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            H, W = img.shape[0:2]
            img = img.reshape((H, W, N, C)).transpose((2, 0, 1, 3))

        if pad_to_target:
            w_pad = (TW - W) if TW is not None else 0
            h_pad = (TH - H) if TH is not None else 0
            if w_pad != 0 or h_pad != 0:
                img = np.pad(img, ((0, 0), (0, h_pad), (0, w_pad), (0, 0)))

        return img, scale

    def pad_to_next_divisor(img, dw=None, dh=None):
        """
        pad image to next divisor of width/height

         dw,dh  int
        """
        _, H, W, _ = img.shape

        w_pad = 0
        if dw is not None:
            w_pad = W % dw
            if w_pad != 0:
                w_pad = dw - w_pad

        h_pad = 0
        if dh is not None:
            h_pad = H % dh
            if h_pad != 0:
                h_pad = dh - h_pad

        if w_pad != 0 or h_pad != 0:
            img = np.pad(img, ((0, 0), (0, h_pad), (0, w_pad), (0, 0)))

        return img

    def np_sigmoid(x: np.ndarray):
        """
        sigmoid with safe check of overflow
        """
        x = -x
        c = x > np.log(np.finfo(x.dtype).max)
        x[c] = 0.0
        result = 1 / (1 + np.exp(x))
        result[c] = 0.0
        return result

    def process_pred(pred, img_w, img_h, anchor):
        pred_h = pred.shape[-3]
        pred_w = pred.shape[-2]
        anchor = np.float32(anchor)[None, :, None, None, :]

        (
            _xv,
            _yv,
        ) = np.meshgrid(
            np.arange(pred_w),
            np.arange(pred_h),
        )
        grid = (
            np.stack((_xv, _yv), 2)
            .reshape((1, 1, pred_h, pred_w, 2))
            .astype(np.float32)
        )

        stride = (img_w // pred_w, img_h // pred_h)

        pred[..., [0, 1, 2, 3, 4]] = np_sigmoid(pred[..., [0, 1, 2, 3, 4]])

        pred[..., 0:2] = (pred[..., 0:2] * 2 - 0.5 + grid) * stride
        pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchor
        return pred

    def predict(img):
        N, C, H, W = img.shape

        # feedforward
        if not args.onnx:
            output = net.predict([img])
        else:
            output = net.run(None, {"in": img})

        # YoloV5Face returns 3x [N,C*16,H,W].
        # C = [cx,cy,w,h,thres, 5*x,y of landmarks, cls_id ]
        # Transpose and cut first 5 channels.
        pred0, pred1, pred2 = [
            pred.reshape((N, C, 16, pred.shape[-2], pred.shape[-1])).transpose(
                0, 1, 3, 4, 2
            )[..., 0:5]
            for pred in output
        ]

        pred0 = process_pred(pred0, W, H, anchor=[[4, 5], [8, 10], [13, 16]]).reshape(
            (1, -1, 5)
        )
        pred1 = process_pred(
            pred1, W, H, anchor=[[23, 29], [43, 55], [73, 105]]
        ).reshape((1, -1, 5))
        pred2 = process_pred(
            pred2, W, H, anchor=[[146, 217], [231, 300], [335, 433]]
        ).reshape((1, -1, 5))

        preds = np.concatenate([pred0, pred1, pred2], 1)[..., :5]
        return preds

    def extract(
        img,
        threshold: float = 0.3,
        fixed_window=0,
        min_face_size=8,
        augment=False,
    ):
        """
        arguments
            img    np.ndarray      ndim 2,3,4

            fixed_window(0)    int  size
                        0 mean don't use
                        fit image in fixed window
                        downscale if bigger than window
                        pad if smaller than window
                        increases performance, but decreases accuracy

            min_face_size(8)

            augment(False)     bool    augment image to increase accuracy
                                    decreases performance

        returns a list of [l,t,r,b] for every batch dimension of img
        """
        H, W, _ = img.shape
        if H > 2048 or W > 2048:
            fixed_window = 2048

        img = img[None, ...]
        if fixed_window != 0:
            fixed_window = max(32, max(1, fixed_window // 32) * 32)
            img, img_scale = fit_in(
                img, fixed_window, fixed_window, pad_to_target=True, allow_upscale=False
            )
        else:
            img = pad_to_next_divisor(img, 64, 64)
            img_scale = 1.0

        _, H, W, _ = img.shape
        img = img.astype(np.float32) / 255.0

        feed_img = img.transpose(0, 3, 1, 2)
        preds = predict(feed_img)

        if augment:
            feed_img = img[:, :, ::-1, :].transpose(0, 3, 1, 2)
            rl_preds = predict(feed_img)
            rl_preds[:, :, 0] = W - rl_preds[:, :, 0]
            preds = np.concatenate([preds, rl_preds], 1)

        faces_per_batch = []
        for pred in preds:
            pred = pred[pred[..., 4] >= threshold]

            x, y, w, h, score = pred.T

            l, t, r, b = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            boxes = np.stack((l, t, r, b), axis=1)
            keep = nms_boxes(boxes, score, 0.5)
            l, t, r, b = l[keep], t[keep], r[keep], b[keep]

            faces = []
            for l, t, r, b in np.stack([l, t, r, b], -1):
                if img_scale != 1.0:
                    l, t, r, b = (
                        l / img_scale,
                        t / img_scale,
                        r / img_scale,
                        b / img_scale,
                    )

                if min(r - l, b - t) < min_face_size:
                    continue
                faces.append((l, t, r, b))

            faces_per_batch.append(faces)

        return faces_per_batch

    return extract


def setup_google_facemesh(net):
    input_height = 192
    input_width = 192

    def extract(img):
        """
        arguments

         img    np.ndarray      HW,HWC,NHWC uint8/float32

        returns (N,468,3)
        """
        H, W, _ = img.shape

        h_scale = H / input_height
        w_scale = W / input_width

        img = resize(img, (input_width, input_height))

        img = img / 255
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        # feedforward
        if not args.onnx:
            output = net.predict([img])
        else:
            output = net.run(None, {"input_1": img})
        lmrks = output[0]

        lmrks = lmrks.reshape((-1, 468, 3))
        lmrks *= (w_scale, h_scale, 1)

        return lmrks

    return extract


def fit_in(img):
    TW = TH = IMG_SIZE
    H, W, _ = img.shape

    SW = W / TW
    SH = H / TH
    scale = 1.0
    if SW > 1.0 or SH > 1.0 or (SW < 1.0 and SH < 1.0):
        scale /= max(SW, SH)

    if scale != 1.0:
        img = cv2.resize(
            img,
            (int(W * scale), int(H * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        H, W = img.shape[0:2]

    w_pad = (TW - W) if TW is not None else 0
    h_pad = (TH - H) if TH is not None else 0
    if w_pad != 0 or h_pad != 0:
        img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)))

    return img


def resize(
    img,
    size: Tuple,
):
    """
    resize to (W,H)
    """
    H, W, _ = img.shape

    TW, TH = size
    if W != TW or H != TH:
        img = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_LINEAR)

    return img


def as_4pts(pts, w_h=None) -> np.ndarray:
    """
    get rect as 4 pts

        0--3
        |  |
        1--2

        w_h(None)    provide (w,h) to scale uniform rect to target size

    returns np.ndarray (4,2) 4 pts with w,h
    """
    if w_h is not None:
        return pts * w_h
    return pts.copy()


def sort_by_area_size(rects: List[np.ndarray]):
    """
    sort list of FRect by largest area descend
    """
    rects = [(rect, polygon_area(as_4pts(rect))) for rect in rects]
    rects = sorted(rects, key=lambda x: x[1], reverse=True)
    rects = [x[0] for x in rects]
    return rects


def face_urect_cut(
    fsi,
    img: np.ndarray,
    coverage: float,
    output_size: int,
    x_offset: float = 0,
    y_offset: float = 0,
):
    """
    Cut the face to square of output_size from img with given coverage using this rect

    returns image,
            uni_mat     uniform matrix to transform uniform img space to uniform cutted space
    """

    uni_rect = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Face rect is not a square, also rect can be rotated

    h, w = img.shape[0:2]

    # Get scaled rect pts to target img
    pts = as_4pts(fsi.face_urect, w_h=(w, h))

    # Estimate transform from global space to local aligned space with bounds [0..1]
    mat = umeyama(pts, uni_rect, True)

    # get corner points in global space
    g_p = transform_points(invert(mat), [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    g_c = g_p[4]

    h_vec = (g_p[1] - g_p[0]).astype(np.float32)
    v_vec = (g_p[3] - g_p[0]).astype(np.float32)

    # calc diagonal vectors between corners in global space
    tb_diag_vec = segment_to_vector(g_p[0], g_p[2]).astype(np.float32)
    bt_diag_vec = segment_to_vector(g_p[3], g_p[1]).astype(np.float32)

    mod = segment_length(g_p[0], g_p[4]) * coverage

    g_c += h_vec * x_offset + v_vec * y_offset

    l_t = np.array(
        [g_c - tb_diag_vec * mod, g_c + bt_diag_vec * mod, g_c + tb_diag_vec * mod],
        np.float32,
    )
    src_pts, dst_pts = l_t, np.float32(
        ((0, 0), (output_size, 0), (output_size, output_size))
    )
    mat = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    src_pts, dst_pts = (l_t / (w, h)).astype(np.float32), np.float32(
        ((0, 0), (1, 0), (1, 1))
    )
    uni_mat = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))

    face_image = cv2.warpAffine(img, mat, (output_size, output_size), cv2.INTER_CUBIC)
    return face_image, uni_mat


def face_ulmrks_transform(face_ulmrks, mat, invert=False) -> "FLandmarks2D":
    """
    Tranforms FLandmarks2D using affine mat and returns new FLandmarks2D()

     mat : np.ndarray
    """
    if invert:
        mat = cv2.invertAffineTransform(mat)

    ulmrks = face_ulmrks.ulmrks.copy()
    ulmrks = np.expand_dims(ulmrks, axis=1)
    ulmrks = cv2.transform(ulmrks, mat, ulmrks.shape).squeeze()

    return FLandmarks2D(type=face_ulmrks.type, ulmrks=ulmrks)


def get_convexhull_mask(
    face_align_ulmrks, h_w, color=(1,), dtype=np.float32
) -> np.ndarray:
    """ """
    h, w = h_w
    ch = len(color)
    lmrks = (face_align_ulmrks.ulmrks * h_w).astype(np.int32)
    mask = np.zeros((h, w, ch), dtype=dtype)
    cv2.fillConvexPoly(mask, cv2.convexHull(lmrks), color)
    return mask


def from_3D_468_landmarks(lmrks):
    """ """
    mat = np.empty((3, 3))
    mat[0, :] = (lmrks[454] - lmrks[234]) / np.linalg.norm(lmrks[454] - lmrks[234])
    mat[1, :] = (lmrks[152] - lmrks[6]) / np.linalg.norm(lmrks[152] - lmrks[6])
    mat[2, :] = np.cross(mat[0, :], mat[1, :])
    pitch, yaw, roll = rotation_matrix_to_euler(mat)

    face_rect = np.array([pitch, yaw * 2, roll], np.float32)
    return face_rect


# ======================
# Main functions
# ======================


def face_detector(
    models,
    tar_img,
    threshold=0.5,
    fixed_window_size=448,
    max_faces=1,
    temporal_smoothing=1,
):
    H, W, _ = tar_img.shape

    extract = models["face_detector"]
    rects = extract(
        tar_img,
        threshold=threshold,
        fixed_window=fixed_window_size,
    )
    rects = rects[0]

    # to list of FaceURect
    rects = [
        np.array([[l, t], [l, b], [r, b], [r, t]], np.float32)
        for l, t, r, b in [[l / W, t / H, r / W, b / H] for l, t, r, b in rects]
    ]

    rects = sort_by_area_size(rects)

    fsi_list = []
    if len(rects) != 0:
        max_faces = max_faces
        if max_faces != 0 and len(rects) > max_faces:
            rects = rects[:max_faces]

        if temporal_smoothing != 1:
            if len(getattr(face_detector, "temporal_rects", [])) != len(rects):
                face_detector.temporal_rects = [[] for _ in range(len(rects))]

        for face_id, face_urect in enumerate(rects):
            if temporal_smoothing != 1:
                if len(face_detector.temporal_rects[face_id]) == 0:
                    face_detector.temporal_rects[face_id].append(as_4pts(face_urect))

                face_detector.temporal_rects[face_id] = face_detector.temporal_rects[
                    face_id
                ][-temporal_smoothing:]
                face_urect = np.mean(face_detector.temporal_rects[face_id], 0)

            if polygon_area(face_urect) != 0:
                fsi_list.append(FaceSwapInfo(face_urect=face_urect))

    return fsi_list


def face_marker(models, frame_image, fsi_list, coverage=1.4, temporal_smoothing=1):
    is_opencv_lbf = False
    is_google_facemesh = True
    is_insightface_2d106 = False

    if temporal_smoothing != 1 and (
        len(getattr(face_marker, "temporal_lmrks", [])) != len(fsi_list)
    ):
        face_marker.temporal_lmrks = [[] for _ in range(len(fsi_list))]

    for face_id, fsi in enumerate(fsi_list):
        if fsi.face_urect is None:
            continue

        # Cut the face to feed to the face marker
        face_image, face_uni_mat = face_urect_cut(
            fsi,
            frame_image,
            coverage,
            (
                256
                if is_opencv_lbf
                else (192 if is_google_facemesh else 192 if is_insightface_2d106 else 0)
            ),
        )
        H, W, _ = face_image.shape

        # if is_opencv_lbf:
        #     lmrks = opencv_lbf(face_image)[0]
        # elif is_google_facemesh:
        #     lmrks = google_facemesh(face_image)[0]
        # elif is_insightface_2d106:
        #     lmrks = insightface_2d106(face_image)[0]
        extract = models["face_marker"]
        lmrks = extract(face_image)[0]

        if temporal_smoothing != 1:
            if len(face_marker.temporal_lmrks[face_id]) == 0:
                face_marker.temporal_lmrks[face_id].append(lmrks)
            face_marker.temporal_lmrks[face_id] = face_marker.temporal_lmrks[face_id][
                -temporal_smoothing:
            ]
            lmrks = np.mean(face_marker.temporal_lmrks[face_id], 0)

        if is_google_facemesh:
            fsi.face_pose = from_3D_468_landmarks(lmrks)

        if is_opencv_lbf:
            lmrks /= (W, H)
        elif is_google_facemesh:
            lmrks = lmrks[..., 0:2] / (W, H)
        elif is_insightface_2d106:
            lmrks = lmrks[..., 0:2] / (W, H)

        face_ulmrks = FLandmarks2D(
            type=(
                ELandmarks2D.L68
                if is_opencv_lbf
                else (
                    ELandmarks2D.L468
                    if is_google_facemesh
                    else ELandmarks2D.L106 if is_insightface_2d106 else None
                )
            ),
            ulmrks=lmrks,
        )
        face_ulmrks = face_ulmrks_transform(face_ulmrks, face_uni_mat, invert=True)
        fsi.face_ulmrks = face_ulmrks

    return fsi_list


def face_aligner(models, frame_image, fsi_list, coverage=2.2, resolution=256):
    head_mode = False
    freeze_z_rotation = False
    align_mode = AlignMode.FROM_RECT
    x_offset = y_offset = 0.0

    for face_id, fsi in enumerate(fsi_list):
        if fsi.face_ulmrks is None:
            continue

        face_ulmrks = fsi.face_ulmrks
        fsi.face_resolution = resolution

        H, W = frame_image.shape[:2]
        if align_mode == AlignMode.FROM_RECT:
            face_align_img, uni_mat = face_urect_cut(
                fsi,
                frame_image,
                coverage=coverage,
                output_size=resolution,
                x_offset=x_offset,
                y_offset=y_offset,
            )
        # elif align_mode == AlignMode.FROM_POINTS:
        #     face_align_img, uni_mat = face_ulmrks.cut(
        #         frame_image,
        #         state.face_coverage + (1.0 if head_mode else 0.0),
        #         state.resolution,
        #         exclude_moving_parts=state.exclude_moving_parts,
        #         head_yaw=head_yaw,
        #         x_offset=state.x_offset,
        #         y_offset=state.y_offset - 0.08 + (-0.50 if head_mode else 0.0),
        #         freeze_z_rotation=freeze_z_rotation,
        #     )
        # elif align_mode == AlignMode.FROM_STATIC_RECT:
        #     rect = FRect.from_ltrb(
        #         [
        #             0.5 - (fsi.face_resolution / W) / 2,
        #             0.5 - (fsi.face_resolution / H) / 2,
        #             0.5 + (fsi.face_resolution / W) / 2,
        #             0.5 + (fsi.face_resolution / H) / 2,
        #         ]
        #     )
        #     face_align_img, uni_mat = rect.cut(
        #         frame_image,
        #         coverage=state.face_coverage,
        #         output_size=state.resolution,
        #         x_offset=state.x_offset,
        #         y_offset=state.y_offset,
        #     )

        fsi.image_to_align_uni_mat = uni_mat
        fsi.face_align_ulmrks = face_ulmrks_transform(face_ulmrks, uni_mat)
        fsi.face_align_image = face_align_img

        # Due to FaceAligner is not well loaded, we can make lmrks mask here
        face_align_lmrks_mask_img = get_convexhull_mask(
            fsi.face_align_ulmrks,
            face_align_img.shape[:2],
            color=(255,),
            dtype=np.uint8,
        )
        fsi.face_align_lmrks_mask = face_align_lmrks_mask_img

    return fsi_list


def face_animator(models, src_img, fsi_list):
    animator_face_id = 0
    relative_power = 0.72

    for i, fsi in enumerate(fsi_list):
        if animator_face_id == i:
            if fsi.face_align_image is None:
                continue

            face_align_image = fsi.face_align_image
            H, W, _ = face_align_image.shape

            net = models["net"]
            if getattr(face_animator, "driving_ref_motion", None) is None:
                face_animator.driving_ref_motion = extract_motion(net, face_align_image)

            anim_image = generate(
                net,
                src_img,
                face_align_image,
                face_animator.driving_ref_motion,
                power=relative_power,
            )
            anim_image = resize(anim_image, (W, H))

            fsi.face_swap_image = anim_image
            break

    return fsi_list


def extract_motion(net, img: np.ndarray):
    in_src = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), np.float32)

    feed_img = normalize_image(
        resize(img, (IMG_SIZE, IMG_SIZE))[..., ::-1], normalize_type="127.5"
    )
    feed_img = feed_img.transpose(2, 0, 1)  # HWC -> CHW
    feed_img = np.expand_dims(feed_img, axis=0)
    feed_img = feed_img.astype(np.float32)

    in_drv_start_motion = np.zeros((1, 20), np.float32)
    in_power = np.zeros((1,), np.float32)

    # feedforward
    if not args.onnx:
        output = net.predict([in_src, feed_img, in_drv_start_motion, in_power])
    else:
        output = net.run(
            None,
            {
                "in_src": in_src,
                "in_drv": feed_img,
                "in_drv_start_motion": in_drv_start_motion,
                "in_power": in_power,
            },
        )
    out_drv_motion = output[1]

    return out_drv_motion


def generate(
    net,
    img_source: np.ndarray,
    img_driver: np.ndarray,
    driver_start_motion: np.ndarray,
    power,
):
    H, W, _ = img_source.shape

    in_src = normalize_image(
        resize(img_source, (IMG_SIZE, IMG_SIZE))[..., :3][..., ::-1],
        normalize_type="127.5",
    )
    in_src = in_src.transpose(2, 0, 1)  # HWC -> CHW
    in_src = np.expand_dims(in_src, axis=0)
    in_src = in_src.astype(np.float32)

    in_drv = normalize_image(
        resize(img_driver, (IMG_SIZE, IMG_SIZE))[..., ::-1], normalize_type="127.5"
    )
    in_drv = in_drv.transpose(2, 0, 1)  # HWC -> CHW
    in_drv = np.expand_dims(in_drv, axis=0)
    in_drv = in_drv.astype(np.float32)

    in_power = np.array([power], np.float32)

    # feedforward
    if not args.onnx:
        output = net.predict([in_src, in_drv, driver_start_motion, in_power])
    else:
        output = net.run(
            None,
            {
                "in_src": in_src,
                "in_drv": in_drv,
                "in_drv_start_motion": driver_start_motion,
                "in_power": in_power,
            },
        )
    out = output[0]

    out = out.transpose(0, 2, 3, 1)[0]

    out += 1.0
    out /= 2.0
    out *= 255.0
    np.clip(out, 0, 255, out=out)

    out_img = out.astype(np.uint8, copy=False)
    out_img = resize(out_img, (W, H))[..., ::-1]

    return out_img


def deepfacelive(models, drv_img, src_img):
    fsi_list = face_detector(
        models,
        drv_img,
        threshold=args.threshold,
        fixed_window_size=args.window_size,
        max_faces=args.max_faces,
        temporal_smoothing=args.temporal_smoothing,
    )
    if len(fsi_list) == 0:
        return None

    fsi_list = face_marker(
        models,
        drv_img,
        fsi_list,
        coverage=args.marker_coverage,
        temporal_smoothing=args.marker_temporal_smoothing,
    )
    fsi_list = face_aligner(
        models,
        drv_img,
        fsi_list,
        coverage=args.face_coverage,
        resolution=args.resolution,
    )
    fsi_list = face_animator(models, src_img, fsi_list)

    aligned_face_id = 0
    for i, fsi in enumerate(fsi_list):
        if aligned_face_id == i:
            aligned_face = fsi.face_align_image
            break

    for fsi in fsi_list:
        swapped_face = fsi.face_swap_image
        if swapped_face is not None:
            break

    return aligned_face, swapped_face


def recognize_from_image(models):
    source_path = args.source
    logger.info("Source: {}".format(source_path))

    src_img = load_image(source_path)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)
    src_img = fit_in(src_img)

    # driver image loop
    for image_path in args.input:
        logger.info("Driving: {}".format(image_path))

        # prepare input data
        drv_img = load_image(image_path)
        drv_img = cv2.cvtColor(drv_img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = deepfacelive(models, drv_img, src_img)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Loggin
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            output = deepfacelive(models, drv_img, src_img)

        aligned_face, swapped_face = output
        if aligned_face is not None and swapped_face is not None:
            view_image = np.concatenate((aligned_face, swapped_face), 1)

            # plot result
            savepath = get_savepath(args.savepath, image_path, ext=".png")
            logger.info(f"saved at : {savepath}")
            cv2.imwrite(savepath, view_image)

    logger.info("Script finished successfully.")


def recognize_from_video(models):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), "Cannot capture source"

    source_path = args.source
    logger.info("Source: {}".format(source_path))

    src_img = load_image(source_path)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)
    src_img = fit_in(src_img)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or not ret:
            break
        if frame_shown and cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) == 0:
            break

        # inference
        output = deepfacelive(models, frame, src_img)

        aligned_face, swapped_face = output
        if aligned_face is not None and swapped_face is not None:
            res_img = np.concatenate((aligned_face, swapped_face), 1)

            # show
            cv2.imshow("frame", res_img)
            frame_shown = True

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(
        WEIGHT_YOLOV5FACE_PATH, MODEL_YOLOV5FACE_PATH, REMOTE_PATH
    )
    check_and_download_models(WEIGHT_FACEMESH_PATH, MODEL_FACEMESH_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        net_face = ailia.Net(
            MODEL_YOLOV5FACE_PATH, WEIGHT_YOLOV5FACE_PATH, env_id=env_id
        )
        net_marker = ailia.Net(MODEL_FACEMESH_PATH, WEIGHT_FACEMESH_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        net = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        net_face = onnxruntime.InferenceSession(
            WEIGHT_YOLOV5FACE_PATH, providers=providers
        )
        net_marker = onnxruntime.InferenceSession(
            WEIGHT_FACEMESH_PATH, providers=providers
        )

    face_detector = setup_yolov5face(net_face)
    face_marker = setup_google_facemesh(net_marker)

    models = {
        "net": net,
        "face_detector": face_detector,
        "face_marker": face_marker,
    }

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == "__main__":
    main()

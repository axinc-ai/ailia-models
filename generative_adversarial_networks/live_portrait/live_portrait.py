import sys
import time

import numpy as np
import cv2
from skimage import transform as sk_trans

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from nms_utils import nms_boxes
from webcamera_utils import get_capture, get_writer  # noqa

from utils_crop import crop_image

# logger
from logging import getLogger  # noqa


# from face_restoration import get_face_landmarks_5
# from face_restoration import align_warp_face, get_inverse_affine
# from face_restoration import paste_faces_to_image

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_DET_PATH = "retinaface_resnet50.onnx"
MODEL_DET_PATH = "retinaface_resnet50.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/gfpgan/"

REALESRGAN_MODEL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

IMAGE_PATH = "s6.jpg"
SAVE_IMAGE_PATH = "output.png"

IMAGE_SIZE = 512

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("LivePortrait", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Model selection
# ======================

WEIGHT_PATH = ".onnx"
MODEL_PATH = ".onnx.prototxt"


# ======================
# Secondary Functions
# ======================


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def face_align(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0

    t1 = sk_trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = sk_trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = sk_trans.SimilarityTransform(rotation=rot)
    t4 = sk_trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)

    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def get_face_analysis(det_face, landmark):

    def get_landmark(img, face):
        input_size = 192

        bbox = face["bbox"]
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = input_size / (max(w, h) * 1.5)
        aimg, M = face_align(img, center, input_size, _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])

        aimg = aimg.transpose(2, 0, 1)  # HWC -> CHW
        aimg = np.expand_dims(aimg, axis=0)
        aimg = aimg.astype(np.float32)

        # feedforward
        if not args.onnx:
            output = landmark.predict([aimg])
        else:
            output = landmark.run(None, {"data": aimg})
        pred = output[0][0]

        pred = pred.reshape((-1, 2))
        pred[:, 0:2] += 1
        pred[:, 0:2] *= input_size[0] // 2

        IM = cv2.invertAffineTransform(M)
        pred = trans_points2d(pred, IM)

        return pred

    def face_analysis(img):
        input_size = 512

        im_ratio = float(img.shape[0]) / img.shape[1]
        if im_ratio > 1:
            new_height = input_size
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        det_img = (det_img - 127.5) / 128
        det_img = det_img.transpose(2, 0, 1)  # HWC -> CHW
        det_img = np.expand_dims(det_img, axis=0)
        det_img = det_img.astype(np.float32)

        # feedforward
        if not args.onnx:
            output = det_face.predict([det_img])
        else:
            output = det_face.run(None, {"input.1": det_img})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        det_thresh = 0.5
        fmc = 3
        feat_stride_fpn = [8, 16, 32]
        center_cache = {}
        for idx, stride in enumerate(feat_stride_fpn):
            scores = output[idx]
            bbox_preds = output[idx + fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = output[idx + fmc * 2] * stride
            height = input_size // stride
            width = input_size // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                num_anchors = 2
                anchor_centers = np.stack(
                    [anchor_centers] * num_anchors, axis=1
                ).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        nms_thresh = 0.4
        keep = nms_boxes(pre_det, [1 for s in pre_det], nms_thresh)
        bboxes = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        if bboxes.shape[0] == 0:
            return []

        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = dict(bbox=bbox, kps=kps, det_score=det_score)
            lmk = get_landmark(img, face)
            face["landmark_2d_106"] = lmk

            ret.append(face)

        src_face = sorted(
            ret,
            key=lambda face: (face["bbox"][2] - face["bbox"][0])
            * (face["bbox"][3] - face["bbox"][1]),
            reverse=True,
        )

        if len(src_face) == 0:
            logger.info("No face detected in the source image.")
            return None
        elif len(src_face) > 1:
            logger.info(
                f"More than one face detected in the image, only pick one face by rule {crop_cfg.direction}."
            )

        src_face = src_face[0]
        lmk = src_face["landmark_2d_106"]  # this is the 106 landmarks from insightface

        return lmk

    return face_analysis


# ======================
# Main functions
# ======================


def preprocess(img):
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    max_dim = 1280
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    division = 2
    new_h = img.shape[0] - (img.shape[0] % division)
    new_w = img.shape[1] - (img.shape[1] % division)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    # img = normalize_image(img, normalize_type="127.5")

    # img = img.transpose(2, 0, 1)  # HWC -> CHW
    # img = np.expand_dims(img, axis=0)
    # img = img.astype(np.float32)

    return img


def post_processing(pred):
    img = pred[0]
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img[:, :, ::-1]  # RGB -> BGR

    img = np.clip(img, -1, 1)
    img = (img + 1) * 127.5
    img = img.astype(np.uint8)

    return img


def crop_src_image(models, img):
    img = img[:, :, ::-1]  # BGR -> RGB
    img = preprocess(img)

    face_analysis = models["face_analysis"]
    lmk = face_analysis(img)

    # crop the face
    crop_info = crop_image(img, lmk, dsize=512, scale=2.3, vy_ratio=-0.125)

    crop_dct = crop_image(img, lmk, dsize=224, scale=1.5, vy_ratio=-0.1)
    img_crop = crop_dct["img_crop"]

    img_crop = img_crop / 255
    img_crop = img_crop.transpose(2, 0, 1)  # HWC -> CHW
    img_crop = np.expand_dims(img_crop, axis=0)
    img_crop = img_crop.astype(np.float32)

    # feedforward
    net = models["landmark_runner"]
    if not args.onnx:
        output = net.predict([img_crop])
    else:
        output = net.run(None, {"input": img_crop})
    out_pts = output[2]

    # 2d landmarks 203 points
    lmk = out_pts[0].reshape(-1, 2) * 224  # scale to 0-224
    # _transform_pts
    M = crop_dct["M_c2o"]
    lmk = lmk @ M[:2, :2].T + M[:2, 2]

    crop_info["lmk_crop"] = lmk
    crop_info["img_crop_256x256"] = cv2.resize(
        crop_info["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
    )
    crop_info["lmk_crop_256x256"] = crop_info["lmk_crop"] * 256 / 512

    return crop_info


def predict(models, crop_info, img):
    source_lmk = crop_info["lmk_crop"]
    img_crop, img_crop_256x256 = crop_info["img_crop"], crop_info["img_crop_256x256"]

    return models


def recognize_from_image(models):
    # prepare input data
    img = load_image(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    crop_info = crop_src_image(models, img)

    if crop_info is None:
        raise Exception("No face detected in the source image!")

    # input image loop
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
                restored_img = predict(models, crop_info, img)
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
            restored_img = predict(models, crop_info, img)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext=".png")
        logger.info(f"saved at : {savepath}")
        cv2.imwrite(savepath, restored_img)

    logger.info("Script finished successfully.")


def recognize_from_video(models):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), "Cannot capture source"

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * args.upscale
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) * args.upscale
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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        restored_img = predict(models, img)
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

        # show
        cv2.imshow("frame", restored_img)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(restored_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info("Script finished successfully.")


def main():
    # # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
    else:
        import onnxruntime

        # init F
        appearance_feature_extractor = onnxruntime.InferenceSession(
            "appearance_feature_extractor.onnx"
        )
        # init M
        motion_extractor = onnxruntime.InferenceSession("motion_extractor.onnx")
        # init W
        warping_module = onnxruntime.InferenceSession("warping_module.onnx")
        # init G
        spade_generator = onnxruntime.InferenceSession("spade_generator.onnx")
        # init S
        stitching = onnxruntime.InferenceSession("stitching.onnx")

        landmark_runner = onnxruntime.InferenceSession("landmark.onnx")

        landmark = onnxruntime.InferenceSession("2d106det.onnx")
        det_face = onnxruntime.InferenceSession("det_10g.onnx")

    face_analysis = get_face_analysis(det_face, landmark)

    models = {
        "appearance_feature_extractor": appearance_feature_extractor,
        "motion_extractor": motion_extractor,
        "warping_module": warping_module,
        "spade_generator": spade_generator,
        "stitching": stitching,
        "landmark_runner": landmark_runner,
        "face_analysis": face_analysis,
    }

    if args.video is not None:
        recognize_from_video(models)
    else:
        recognize_from_image(models)


if __name__ == "__main__":
    main()

import sys
import time
from logging import getLogger

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

from .nuscenes_dataset import NuScenesDataset


logger = getLogger(__name__)

# from sample_utils import decode_batch, mask_to_bboxes, draw_bbox

# ======================
# Parameters
# ======================

# WEIGHT_PATH = "bev_encoder.onnx"
# MODEL_PATH = "bev_encoder.onnx.prototxt"
WEIGHT_PATH = "bev_encoder.msfix.onnx"
MODEL_PATH = "bev_encoder.msfix.onnx.prototxt"
WEIGHT_XXX_PATH = "track_head.onnx"
MODEL_XXX_PATH = "track_head.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/xxx/"

IMAGE_PATH = "nuscenes"
SAVE_IMAGE_PATH = "output.png"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

THRESHOLD = 0.4
IOU = 0.45

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("UniAD", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================

# def draw_bbox(img, bboxes):
#     return img


# ======================
# Main functions
# ======================


def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape

    # adaptive_resize
    scale = h / min(im_h, im_w)
    ow, oh = int(im_w * scale), int(im_h * scale)
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BICUBIC))

    # resize
    IMAGE_SIZE = 224
    short, long = (im_w, im_h) if im_w <= im_h else (im_h, im_w)
    new_short, new_long = IMAGE_SIZE, (IMAGE_SIZE * long) // short
    ow, oh = (new_short, new_long) if im_w <= im_h else (new_long, new_short)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.Resampling.BICUBIC))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x : x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y : y + h, :, :]

    # center_crop
    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = (oh - IMAGE_SIZE) // 2
    bottom = top + IMAGE_SIZE
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (ow - IMAGE_SIZE) // 2
    right = left + IMAGE_SIZE
    if top >= 0 and bottom <= oh and left >= 0 and right <= ow:
        img = img[top:bottom, left:right, :]
    else:
        # If the image is too small, pad it with zeros
        pad_h = max(IMAGE_SIZE, oh)
        pad_w = max(IMAGE_SIZE, ow)
        pad_img = np.zeros((pad_h, pad_w, 3))

        top_pad = (pad_h - oh) // 2
        left_pad = (pad_w - ow) // 2
        pad_img[top_pad : top_pad + oh, left_pad : left_pad + ow, :] = img
        img = pad_img

    img = normalize_image(img, normalize_type="ImageNet")

    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def predict(models, img):
    # shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    # img = preprocess(img, shape)
    # img = np.load("img_0.npy")
    # can_bus = np.load("can_bus_0.npy")
    # lidar2img = np.load("lidar2img_0.npy")
    # img_shape = np.load("img_shape_0.npy")
    # prev_bev = np.load("prev_bev_0.npy")
    img = np.load("img_1.npy")
    can_bus = np.load("can_bus_1.npy")
    lidar2img = np.load("lidar2img_1.npy")
    img_shape = np.load("img_shape_1.npy")
    prev_bev = np.load("prev_bev_1.npy")

    # bev_encoder = models["bev_encoder"]

    # # feedforward
    # if not args.onnx:
    #     output = bev_encoder.predict([img, can_bus, lidar2img, img_shape, prev_bev])
    # else:
    #     output = bev_encoder.run(
    #         None,
    #         {
    #             "img": img,
    #             "can_bus": can_bus,
    #             "lidar2img": lidar2img,
    #             "img_shape": img_shape,
    #             "prev_bev": prev_bev,
    #         },
    #     )
    # bev_embed, bev_pos = output

    # print("bev_embed---", bev_embed.shape)
    # print(bev_embed)
    # print("bev_pos---", bev_pos.shape)
    # print(bev_pos)

    bev_embed = np.load("bev_embed_3.npy")
    query = np.load("query_3.npy")
    ref_pts = np.load("ref_pts_3.npy")

    track_head = models["track_head"]

    # feedforward
    if not args.onnx:
        output = track_head.predict([img, bev_embed, query, ref_pts])
    else:
        output = track_head.run(
            None,
            {
                "bev_embed": bev_embed,
                "query": query,
                "ref_pts": ref_pts,
            },
        )
    # bev_embed, bev_pos = output

    return output


def recognize_from_image(models):
    ann_file = "data/infos/nuscenes_infos_temporal_val.pkl"

    info = dict(
        cams=dict(
            CAM_FRONT=dict(
                data_path="",
                lidar2img_path="",
                img_shape_path="",
            ),
            CAM_FRONT_RIGHT=dict(
                data_path="",
                lidar2img_path="",
                img_shape_path="",
            ),
            CAM_FRONT_LEFT=dict(
                data_path="",
                lidar2img_path="",
                img_shape_path="",
            ),
            CAM_BACK=dict(
                data_path="",
                lidar2img_path="",
                img_shape_path="",
            ),
            CAM_BACK_LEFT=dict(
                data_path="",
                lidar2img_path="",
                img_shape_path="",
            ),
            CAM_BACK_RIGHT=dict(
                data_path="",
                lidar2img_path="",
                img_shape_path="",
            ),
        )
    )

    # img_filename = [
    #     "samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281446912460.jpg",
    #     "samples/CAM_FRONT_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281446920339.jpg",
    #     "samples/CAM_FRONT_LEFT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281446904854.jpg",
    #     "samples/CAM_BACK/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281446937525.jpg",
    #     "samples/CAM_BACK_LEFT/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281446947423.jpg",
    #     "samples/CAM_BACK_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281446927893.jpg",
    # ]
    img_filename = [
        "samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg",
        "samples/CAM_FRONT_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281439770339.jpg",
        "samples/CAM_FRONT_LEFT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg",
        "samples/CAM_BACK/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281439787525.jpg",
        "samples/CAM_BACK_LEFT/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281439797423.jpg",
        "samples/CAM_BACK_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281439777893.jpg",
    ]

    dataset = NuScenesDataset(
        **{
            "type": "NuScenesE2EDataset",
            "data_root": "data/nuscenes/",
            "ann_file": "data/infos/nuscenes_infos_temporal_val.pkl",
            "pipeline": [
                {
                    "type": "LoadMultiViewImageFromFilesInCeph",
                    "to_float32": True,
                    "file_client_args": {"backend": "disk"},
                    "img_root": "data/nuscenes/",
                },
                {
                    "type": "NormalizeMultiviewImage",
                    "mean": [103.53, 116.28, 123.675],
                    "std": [1.0, 1.0, 1.0],
                    "to_rgb": False,
                },
                {"type": "PadMultiViewImage", "size_divisor": 32},
                {
                    "type": "LoadAnnotations3D_E2E",
                    "with_bbox_3d": False,
                    "with_label_3d": False,
                    "with_attr_label": False,
                    "with_future_anns": True,
                    "with_ins_inds_3d": False,
                    "ins_inds_add_1": True,
                },
                {
                    "type": "GenerateOccFlowLabels",
                    "grid_conf": {
                        "xbound": [-50.0, 50.0, 0.5],
                        "ybound": [-50.0, 50.0, 0.5],
                        "zbound": [-10.0, 10.0, 20.0],
                    },
                    "ignore_index": 255,
                    "only_vehicle": True,
                    "filter_invisible": False,
                },
                {
                    "type": "MultiScaleFlipAug3D",
                    "img_scale": (1600, 900),
                    "pts_scale_ratio": 1,
                    "flip": False,
                    "transforms": [
                        {
                            "type": "DefaultFormatBundle3D",
                            "class_names": [
                                "car",
                                "truck",
                                "construction_vehicle",
                                "bus",
                                "trailer",
                                "barrier",
                                "motorcycle",
                                "bicycle",
                                "pedestrian",
                                "traffic_cone",
                            ],
                            "with_label": False,
                        },
                        {
                            "type": "CustomCollect3D",
                            "keys": [
                                "img",
                                "timestamp",
                                "l2g_r_mat",
                                "l2g_t",
                                "gt_lane_labels",
                                "gt_lane_bboxes",
                                "gt_lane_masks",
                                "gt_segmentation",
                                "gt_instance",
                                "gt_centerness",
                                "gt_offset",
                                "gt_flow",
                                "gt_backward_flow",
                                "gt_occ_has_invalid_frame",
                                "gt_occ_img_is_valid",
                                "sdc_planning",
                                "sdc_planning_mask",
                                "command",
                            ],
                        },
                    ],
                },
            ],
            "classes": [
                "car",
                "truck",
                "construction_vehicle",
                "bus",
                "trailer",
                "barrier",
                "motorcycle",
                "bicycle",
                "pedestrian",
                "traffic_cone",
            ],
            "modality": {
                "use_lidar": False,
                "use_camera": True,
                "use_radar": False,
                "use_map": False,
                "use_external": True,
            },
            "test_mode": True,
            "box_type_3d": "LiDAR",
            "file_client_args": {"backend": "disk"},
            "patch_size": [102.4, 102.4],
            "canvas_size": (200, 200),
            "bev_size": (200, 200),
            "predict_steps": 12,
            "past_steps": 4,
            "fut_steps": 4,
            "occ_n_future": 6,
            "use_nonlinear_optimizer": True,
            "eval_mod": ["det", "map", "track", "motion"],
        }
    )

    data_loader = []
    for i, data in enumerate(data_loader):
        result = predict(models, None)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_XXX_PATH, MODEL_XXX_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        bev_encoder = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        track_head = ailia.Net(MODEL_XXX_PATH, WEIGHT_XXX_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        bev_encoder = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        track_head = onnxruntime.InferenceSession(WEIGHT_XXX_PATH, providers=providers)

    models = dict(
        bev_encoder=bev_encoder,
        track_head=track_head,
    )

    recognize_from_image(models)


if __name__ == "__main__":
    main()

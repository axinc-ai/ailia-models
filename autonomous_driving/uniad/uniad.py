import sys
import time
from logging import getLogger

import numpy as np
from torch.utils.data import DataLoader

import ailia


# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

from nuscenes_dataset import NuScenesDataset
from track_instance import Instances


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


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("UniAD", IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================


def empty_tracks():
    track_instances = Instances((1, 1))
    query = np.load("query_embedding.npy")
    num_queries, dim = query.shape
    reference_points_weight = np.load("reference_points_weight.npy")
    reference_points_bias = np.load("reference_points_bias.npy")
    track_instances.ref_pts = (
        query[..., : dim // 2] @ reference_points_weight.T + reference_points_bias
    )

    track_instances.query = query

    track_instances.obj_idxes = np.full((len(track_instances)), -1, dtype=np.int64)

    return track_instances


# ======================
# Main functions
# ======================


def predict(models, img, img_metas, prev_bev=None, track_instances=None):
    can_bus = img_metas["can_bus"].astype(np.float32)
    lidar2img = img_metas["lidar2img"].astype(np.float32)
    img_shape = img_metas["img_shape"]
    if prev_bev is None:
        prev_bev = np.zeros((40000, 1, 256), dtype=np.float32)

    # feedforward
    bev_encoder = models["bev_encoder"]
    if not args.onnx:
        output = bev_encoder.predict([img, can_bus, lidar2img, img_shape, prev_bev])
    else:
        output = bev_encoder.run(
            None,
            {
                "img": img,
                "can_bus": can_bus,
                "lidar2img": lidar2img,
                "img_shape": img_shape,
                "prev_bev": prev_bev,
            },
        )
    bev_embed, bev_pos = output

    query = track_instances.query
    ref_pts = track_instances.ref_pts

    # feedforward
    track_head = models["track_head"]
    if not args.onnx:
        output = track_head.predict([bev_embed, query, ref_pts])
    else:
        output = track_head.run(
            None,
            {
                "bev_embed": bev_embed,
                "query": query,
                "ref_pts": ref_pts,
            },
        )
    (
        output_classes,
        output_coords,
        last_ref_pts,
        query_feats,
        all_past_traj_preds,
    ) = output

    out = {
        "pred_logits": output_classes,
        "pred_boxes": output_coords,
        "ref_pts": last_ref_pts,
        "bev_embed": bev_embed,
        "query_embeddings": query_feats,
        "all_past_traj_preds": all_past_traj_preds,
        "bev_pos": bev_pos,
    }

    return out


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
            "ann_file": ann_file,
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

    def collate_fn(batch):
        return batch

    data_loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
    )

    track_instances = empty_tracks()

    for i, data in enumerate(data_loader):
        img = data[0]["img"]
        img_metas = data[0]["img_metas"]
        img = np.stack(img).transpose(0, 3, 1, 2)  # N, C, H, W
        img = np.expand_dims(img, axis=0)  # B, N, C, H, W

        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]
        track_instances = Instances.cat([other_inst, active_inst])

        output = predict(models, img, img_metas, track_instances=track_instances)

        # Apply sigmoid and get max values along last axis
        logits = output["pred_logits"][-1, 0, :]
        sigmoid_logits = 1 / (1 + np.exp(-logits))
        track_scores = np.max(sigmoid_logits, axis=-1)

        pass

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

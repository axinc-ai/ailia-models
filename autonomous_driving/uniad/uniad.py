import glob
import os
import sys
import time
from logging import getLogger

import cv2
import numpy as np
from torch.utils.data import DataLoader

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from math_utils import sigmoid

from nuscenes_dataset import NuScenesDataset
from track_instance import Instances
from tracker import RuntimeTrackerBase


logger = getLogger(__name__)


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
    track_instances.matched_gt_idxes = np.full(
        (len(track_instances)), -1, dtype=np.int64
    )
    track_instances.disappear_time = np.zeros((len(track_instances)), dtype=np.int64)

    track_instances.iou = np.zeros((len(track_instances),), dtype=np.float32)
    track_instances.scores = np.zeros((len(track_instances),), dtype=np.float32)
    track_instances.track_scores = np.zeros((len(track_instances),), dtype=np.float32)

    # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
    num_box_dims = 10
    track_instances.pred_boxes = np.zeros(
        (len(track_instances), num_box_dims), dtype=np.float32
    )

    num_classes = 10  # number of classes from the dataset config
    track_instances.pred_logits = np.zeros(
        (len(track_instances), num_classes), dtype=np.float32
    )

    mem_bank_len = 4  # typical memory bank length
    track_instances.mem_bank = np.zeros(
        (len(track_instances), mem_bank_len, dim // 2), dtype=np.float32
    )
    track_instances.mem_padding_mask = np.ones(
        (len(track_instances), mem_bank_len), dtype=bool
    )
    track_instances.save_period = np.zeros((len(track_instances),), dtype=np.float32)

    return track_instances


def decode_single(
    cls_scores, bbox_preds, track_scores, obj_idxes, with_mask=True, img_metas=None
):
    """Decode bboxes.
    Args:
        cls_scores (Tensor): Outputs from the classification head, \
            shape [num_query, cls_out_channels]. Note \
            cls_out_channels should includes background.
        bbox_preds (Tensor): Outputs from the regression \
            head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
            Shape [num_query, 9].

    Returns:
        list[dict]: Decoded boxes.
    """
    max_num = self.max_num
    max_num = min(cls_scores.size(0), self.max_num)

    cls_scores = cls_scores.sigmoid()
    _, indexs = cls_scores.max(dim=-1)
    labels = indexs % self.num_classes

    _, bbox_index = track_scores.topk(max_num)

    labels = labels[bbox_index]
    bbox_preds = bbox_preds[bbox_index]
    track_scores = track_scores[bbox_index]
    obj_idxes = obj_idxes[bbox_index]

    scores = track_scores

    final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
    final_scores = track_scores
    final_preds = labels

    # use score threshold
    if self.score_threshold is not None:
        thresh_mask = final_scores > self.score_threshold

    if self.with_nms:
        boxes_for_nms = xywhr2xyxyr(
            img_metas[0]["box_type_3d"](final_box_preds[:, :], 9).bev
        )
        nms_mask = boxes_for_nms.new_zeros(boxes_for_nms.shape[0]) > 0
        # print(self.nms_iou_thres)
        try:
            selected = nms_bev(boxes_for_nms, final_scores, thresh=self.nms_iou_thres)
            nms_mask[selected] = True
        except:
            print("Error", boxes_for_nms, final_scores)
            nms_mask = boxes_for_nms.new_ones(boxes_for_nms.shape[0]) > 0
    if self.post_center_range is not None:
        self.post_center_range = torch.tensor(
            self.post_center_range, device=scores.device
        )
        mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
        mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

        if self.score_threshold:
            mask &= thresh_mask
        if not with_mask:
            mask = torch.ones_like(mask) > 0
        if self.with_nms:
            mask &= nms_mask

        boxes3d = final_box_preds[mask]
        scores = final_scores[mask]
        labels = final_preds[mask]
        track_scores = track_scores[mask]
        obj_idxes = obj_idxes[mask]
        predictions_dict = {
            "bboxes": boxes3d,
            "scores": scores,
            "labels": labels,
            "track_scores": track_scores,
            "obj_idxes": obj_idxes,
            "bbox_index": bbox_index,
            "mask": mask,
        }

    else:
        raise NotImplementedError(
            "Need to reorganize output as a batch, only "
            "support post_center_range is not None for now!"
        )
    return predictions_dict


def det_instances2results(instances, results, img_metas):
    """
    Outs:
    active_instances. keys:
    - 'pred_logits':
    - 'pred_boxes': normalized bboxes
    - 'scores'
    - 'obj_idxes'
    out_dict. keys:
        - boxes_3d (torch.Tensor): 3D boxes.
        - scores (torch.Tensor): Prediction scores.
        - labels_3d (torch.Tensor): Box labels.
        - attrs_3d (torch.Tensor, optional): Box attributes.
        - track_ids
        - tracking_score
    """
    # filter out sleep querys
    if instances.pred_logits.numel() == 0:
        return [None]
    bbox_dict = dict(
        cls_scores=instances.pred_logits,
        bbox_preds=instances.pred_boxes,
        track_scores=instances.scores,
        obj_idxes=instances.obj_idxes,
    )
    # bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
    # decode
    with_mask = True
    all_cls_scores = bbox_dict["cls_scores"]
    all_bbox_preds = bbox_dict["bbox_preds"]
    track_scores = bbox_dict["track_scores"]
    obj_idxes = bbox_dict["obj_idxes"]

    batch_size = all_cls_scores.size()[0]
    predictions_list = []
    # bs size = 1
    predictions_list.append(
        decode_single(
            all_cls_scores,
            all_bbox_preds,
            track_scores,
            obj_idxes,
            with_mask,
            img_metas,
        )
    )

    bboxes_dict = predictions_list[0]
    bboxes = bboxes_dict["bboxes"]
    bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
    labels = bboxes_dict["labels"]
    scores = bboxes_dict["scores"]

    track_scores = bboxes_dict["track_scores"]
    obj_idxes = bboxes_dict["obj_idxes"]
    result_dict = results[0]
    result_dict_det = dict(
        boxes_3d_det=bboxes.to("cpu"),
        scores_3d_det=scores.cpu(),
        labels_3d_det=labels.cpu(),
    )
    if result_dict is not None:
        result_dict.update(result_dict_det)
    else:
        result_dict = None

    return [result_dict]


def to_video(folder_path, out_path, fps=4, downsample=1):
    imgs_path = glob.glob(os.path.join(folder_path, "*.jpg"))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(
            img,
            (width // downsample, height // downsample),
            interpolation=cv2.INTER_AREA,
        )
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


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

    score_thresh = 0.4
    filter_score_thresh = 0.35
    miss_tolerance = 5
    track_base = RuntimeTrackerBase(
        score_thresh=score_thresh,
        filter_score_thresh=filter_score_thresh,
        miss_tolerance=miss_tolerance,
    )

    track_instances = empty_tracks()

    bbox_results = []
    for i, data in enumerate(data_loader):
        img = data[0]["img"]
        img_metas = data[0]["img_metas"]
        img = np.stack(img).transpose(0, 3, 1, 2)  # N, C, H, W
        img = np.expand_dims(img, axis=0)  # B, N, C, H, W

        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]
        track_instances = Instances.cat([other_inst, active_inst])

        output = predict(models, img, img_metas, track_instances=track_instances)
        output_classes = output["pred_logits"]
        output_coords = output["pred_boxes"]
        last_ref_pts = output["ref_pts"]
        query_feats = output["query_embeddings"]

        # Apply sigmoid and get max values along last axis
        logits = output_classes[-1, 0, :]
        sigmoid_logits = sigmoid(logits)
        track_scores = np.max(sigmoid_logits, axis=-1)

        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query
        track_instances.obj_idxes[900] = -2

        track_base.update(track_instances, None)

        frame_res = output
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]

        results = [dict()]
        get_keys = [
            "bev_embed",
            "bev_pos",
            "track_query_embeddings",
            "track_bbox_results",
            "boxes_3d",
            "scores_3d",
            "labels_3d",
            "track_scores",
            "track_ids",
            "sdc_boxes_3d",
            "sdc_scores_3d",
            "sdc_track_scores",
            "sdc_track_bbox_results",
            "sdc_embedding",
        ]
        results[0].update({k: frame_res[k] for k in get_keys})
        result_track = det_instances2results(track_instances_fordet, results, img_metas)

        result = [dict() for i in range(len(img_metas))]

        # seg_head
        # result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)
        result_seg = [dict(pts_bbox=None, ret_iou=None, args_tuple=None)]

        for i, res in enumerate(result):
            res["token"] = img_metas[i]["sample_idx"]
            res.update(result_track[i])
            # if self.with_motion_head:
            #     res.update(result_motion[i])
            # if self.with_seg_head:
            #     res.update(result_seg[i])

        batch_size = len(result)
        bbox_results.extend(result)

    ret_results = dict()
    ret_results["bbox_results"] = bbox_results

    to_video("output_folder", "output_video.avi")

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

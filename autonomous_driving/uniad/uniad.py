import glob
import os
import sys
import time
from logging import getLogger

import ailia
import cv2
import numpy as np
from nuscenes import NuScenes
from torch.utils.data import DataLoader

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, get_savepath, update_parser
from math_utils import sigmoid
from model_utils import check_and_download_models

# import local modules
from lidar_box3d import LiDARInstance3DBoxes
from nuscenes_dataset import NuScenesDataset
from render import BEVRender, CameraRender
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
WEIGHT_TRACK_HEAD_PATH = "track_head.onnx"
MODEL_TRACK_HEAD_PATH = "track_head.onnx.prototxt"
WEIGHT_MEMORY_BANK_PATH = "memory_bank.onnx"
MODEL_MEMORY_BANK_PATH = "memory_bank.onnx.prototxt"
WEIGHT_MEMORY_BANK_UPD_PATH = "memory_bank_update.onnx"
MODEL_MEMORY_BANK_UPD_PATH = "memory_bank_update.onnx.prototxt"
WEIGHT_QUERY_INTERACTION_PATH = "query_interact.onnx"
MODEL_QUERY_INTERACTION_PATH = "query_interact.onnx.prototxt"
WEIGHT_SEG_HEAD_PATH = "seg_head.onnx"
MODEL_SEG_HEAD_PATH = "seg_head.onnx.prototxt"
WEIGHT_MOTION_HEAD_PATH = "motion_head.onnx"
MODEL_MOTION_HEAD_PATH = "motion_head.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/uniad/"

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


class AgentPredictionData:
    """
    Agent data class, includes bbox, traj, and occflow
    """

    def __init__(
        self,
        pred_score,
        pred_label,
        pred_center,
        pred_dim,
        pred_yaw,
        pred_vel,
        pred_traj,
        pred_traj_score,
        pred_track_id=None,
        pred_occ_map=None,
        is_sdc=False,
        past_pred_traj=None,
        command=None,
        attn_mask=None,
    ):
        self.pred_score = pred_score
        self.pred_label = pred_label
        self.pred_center = pred_center
        self.pred_dim = pred_dim
        # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, maybe we should change this. [DONE]
        # self.pred_yaw = pred_yaw
        self.pred_yaw = -pred_yaw - np.pi / 2
        self.pred_vel = pred_vel
        self.pred_traj = pred_traj
        self.pred_traj_score = pred_traj_score
        self.pred_track_id = pred_track_id
        self.pred_occ_map = pred_occ_map
        if self.pred_traj is not None:
            if isinstance(self.pred_traj_score, int):
                self.pred_traj_max = self.pred_traj
            else:
                self.pred_traj_max = self.pred_traj[self.pred_traj_score.argmax()]
        else:
            self.pred_traj_max = None
        self.nusc_box = Box(
            center=pred_center,
            size=pred_dim,
            orientation=Quaternion(axis=[0, 0, 1], radians=self.pred_yaw),
            label=pred_label,
            score=pred_score,
        )
        if is_sdc:
            self.pred_center = [0, 0, -1.2 + 1.56 / 2]
        self.is_sdc = is_sdc
        self.past_pred_traj = past_pred_traj
        self.command = command
        self.attn_mask = attn_mask


class Visualizer:
    def __init__(
        self,
        bbox_results,
        dataroot="data/nuscenes",
        version="v1.0-mini",
        show_gt_boxes=False,
    ):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

        self.token_set = set()
        self.predictions = self._parse_predictions(bbox_results)
        self.bev_render = BEVRender(show_gt_boxes=show_gt_boxes)
        self.cam_render = CameraRender(show_gt_boxes=show_gt_boxes)

    def _parse_predictions(self, bbox_results):
        outputs = bbox_results
        prediction_dict = dict()
        for k in range(len(outputs)):
            token = outputs[k]["token"]
            self.token_set.add(token)

            # detection
            bboxes: LiDARInstance3DBoxes = outputs[k]["boxes_3d"]
            scores = outputs[k]["scores_3d"]
            labels = outputs[k]["labels_3d"]

            track_scores = scores
            track_labels = labels
            track_boxes = bboxes.tensor

            track_centers = bboxes.gravity_center
            track_dims = bboxes.dims
            track_yaw = bboxes.yaw

            track_ids = outputs[k]["track_ids"]

            # speed
            track_velocity = bboxes.tensor[:, -2:]

            # trajectories
            trajs = outputs[k]["traj"]
            traj_scores = outputs[k]["traj_scores"]

            predicted_agent_list = []
            for i in range(track_scores.shape[0]):
                if track_scores[i] < 0.25:
                    continue

                if i < len(track_ids):
                    track_id = track_ids[i]
                else:
                    track_id = 0

                predicted_agent_list.append(
                    AgentPredictionData(
                        track_scores[i],
                        track_labels[i],
                        track_centers[i],
                        track_dims[i],
                        track_yaw[i],
                        track_velocity[i],
                        trajs[i],
                        traj_scores[i],
                        pred_track_id=track_id,
                        pred_occ_map=None,
                        past_pred_traj=None,
                    )
                )

            # detection
            bboxes = outputs[k]["sdc_boxes_3d"]
            scores = outputs[k]["sdc_scores_3d"]
            labels = 0

            track_scores = scores
            track_labels = labels

            track_centers = bboxes.gravity_center
            track_dims = bboxes.dims
            track_yaw = bboxes.yaw
            track_velocity = bboxes.tensor[:, -2:]

            command = outputs[k]["command"][0]
            planning_agent = AgentPredictionData(
                track_scores[0],
                track_labels,
                track_centers[0],
                track_dims[0],
                track_yaw[0],
                track_velocity[0],
                outputs[k]["planning_traj"][0],
                1,
                pred_track_id=-1,
                pred_occ_map=None,
                past_pred_traj=None,
                is_sdc=True,
                command=command,
            )
            predicted_agent_list.append(planning_agent)

            prediction_dict[token] = dict(
                predicted_agent_list=predicted_agent_list,
                # predicted_map_seg=None,
                predicted_planning=planning_agent,
            )
        return prediction_dict

    def visualize_bev(self, sample_token):
        self.bev_render.render_pred_box_data(
            self.predictions[sample_token]["predicted_agent_list"]
        )
        self.bev_render.render_pred_traj(
            self.predictions[sample_token]["predicted_agent_list"]
        )
        self.bev_render.render_pred_box_data(
            [self.predictions[sample_token]["predicted_planning"]]
        )
        self.bev_render.render_planning_data(
            self.predictions[sample_token]["predicted_planning"],
            show_command=self.show_command,
        )
        self.bev_render.render_sdc_car()
        self.bev_render.render_legend()

        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)

    def visualize_cam(self, sample_token, out_filename):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)
        # self.cam_render.render_pred_track_bbox(
        #     self.predictions[sample_token]["predicted_agent_list"],
        #     sample_token,
        #     self.nusc,
        # )
        # self.cam_render.render_pred_traj(
        #     self.predictions[sample_token]["predicted_agent_list"],
        #     sample_token,
        #     self.nusc,
        #     render_sdc=self.with_planning,
        # )
        self.cam_render.save_fig(out_filename + "_cam.jpg")


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


def denormalize_bbox(normalized_bboxes):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = np.arctan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = np.exp(w)
    l = np.exp(l)
    h = np.exp(h)

    # velocity
    vx = normalized_bboxes[:, 8:9]
    vy = normalized_bboxes[:, 9:10]
    denormalized_bboxes = np.concatenate([cx, cy, cz, w, l, h, rot, vx, vy], axis=-1)

    return denormalized_bboxes


def decode_single(cls_scores, bbox_preds, track_scores, obj_idxes, with_mask=True):
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
    max_num = 300
    max_num = min(cls_scores.shape[0], max_num)

    num_classes = 10
    cls_scores = sigmoid(cls_scores)
    indexs = np.argmax(cls_scores, axis=-1)
    labels = indexs % num_classes

    bbox_index = np.argpartition(track_scores, -max_num)[-max_num:]
    bbox_index = bbox_index[np.argsort(track_scores[bbox_index])[::-1]]

    labels = labels[bbox_index]
    bbox_preds = bbox_preds[bbox_index]
    track_scores = track_scores[bbox_index]
    obj_idxes = obj_idxes[bbox_index]

    scores = track_scores

    final_box_preds = denormalize_bbox(bbox_preds)
    final_scores = track_scores
    final_preds = labels

    post_center_range = np.array(
        [-61.2000, -61.2000, -10.0000, 61.2000, 61.2000, 10.0000]
    )
    mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
    mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)
    if not with_mask:
        mask = np.ones_like(mask) > 0

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

    return predictions_dict


def track_instances2results(track_instances, with_mask=True):
    all_cls_scores = track_instances.pred_logits
    all_bbox_preds = track_instances.pred_boxes
    track_scores = track_instances.scores
    obj_idxes = track_instances.obj_idxes
    bboxes_dict = decode_single(
        all_cls_scores,
        all_bbox_preds,
        track_scores,
        obj_idxes,
        with_mask,
    )

    bboxes = bboxes_dict["bboxes"]
    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
    labels = bboxes_dict["labels"]
    scores = bboxes_dict["scores"]
    bbox_index = bboxes_dict["bbox_index"]
    track_scores = bboxes_dict["track_scores"]
    obj_idxes = bboxes_dict["obj_idxes"]
    result_dict = dict(
        boxes_3d=bboxes,
        scores_3d=scores,
        labels_3d=labels,
        track_scores=track_scores,
        bbox_index=bbox_index,
        track_ids=obj_idxes,
        mask=bboxes_dict["mask"],
        track_bbox_results=[
            [
                bboxes,
                scores,
                labels,
                bbox_index,
                bboxes_dict["mask"],
            ]
        ],
    )
    return result_dict


def det_instances2results(instances, result_dict):
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
    if instances.pred_logits.size == 0:
        return [None]

    # decode
    with_mask = True
    all_cls_scores = instances.pred_logits
    all_bbox_preds = instances.pred_boxes
    track_scores = instances.scores
    obj_idxes = instances.obj_idxes
    bboxes_dict = decode_single(
        all_cls_scores,
        all_bbox_preds,
        track_scores,
        obj_idxes,
        with_mask,
    )

    bboxes = bboxes_dict["bboxes"]
    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
    labels = bboxes_dict["labels"]
    scores = bboxes_dict["scores"]

    track_scores = bboxes_dict["track_scores"]
    obj_idxes = bboxes_dict["obj_idxes"]
    result_dict.update(
        dict(
            boxes_3d_det=bboxes,
            scores_3d_det=scores,
            labels_3d_det=labels,
        )
    )

    return result_dict


def get_trajs(preds_dicts, bbox_results):
    """
    Generates trajectories from the prediction results, bounding box results.
    """
    num_samples = len(bbox_results)
    num_layers = preds_dicts["all_traj_preds"].shape[0]
    ret_list = []
    for i in range(num_samples):
        preds = dict()
        for j in range(num_layers):
            subfix = "_" + str(j) if j < (num_layers - 1) else ""
            traj = preds_dicts["all_traj_preds"][j, i]
            traj_scores = preds_dicts["all_traj_scores"][j, i]

            preds["traj" + subfix] = traj
            preds["traj_scores" + subfix] = traj_scores
        ret_list.append(preds)
    return ret_list


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

    # print("bev_embed---", bev_embed.shape)
    # print(bev_embed)
    # print("bev_pos---", bev_pos.shape)
    # print(bev_pos)

    # bev_embed = np.load("bev_embed_3.npy")
    # query = np.load("query_3.npy")
    # ref_pts = np.load("ref_pts_3.npy")

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


def memory_bank(models, track_instances):
    key_padding_mask = track_instances.mem_padding_mask  # [n_, memory_bank_len]
    valid_idxes = key_padding_mask[:, -1] == 0
    embed = track_instances.output_embedding[valid_idxes]  # (n, 256)

    if 0 < len(embed):
        net = models["memory_bank"]
        if not args.onnx:
            output = net.predict(
                [
                    track_instances.mem_padding_mask,
                    track_instances.output_embedding,
                    track_instances.mem_bank,
                ]
            )
        else:
            output = net.run(
                None,
                {
                    "mem_padding_mask": track_instances.mem_padding_mask,
                    "output_embedding": track_instances.output_embedding,
                    "mem_bank": track_instances.mem_bank,
                },
            )
        track_instances.output_embedding = output[0]

    embed = track_instances.output_embedding[:, None]  # ( N, 1, 256)
    save_period = track_instances.save_period
    saved_idxes = (save_period == 0) & (track_instances.scores > 0)
    saved_embed = embed[saved_idxes]

    if 0 < len(saved_embed):
        net = models["memory_bank_update"]
        if not args.onnx:
            output = net.predict(
                [
                    track_instances.output_embedding,
                    track_instances.scores,
                    track_instances.mem_padding_mask,
                    track_instances.save_period,
                    track_instances.mem_bank,
                ]
            )
        else:
            output = net.run(
                None,
                {
                    "output_embedding": track_instances.output_embedding,
                    "scores": track_instances.scores,
                    "mem_padding_mask": track_instances.mem_padding_mask,
                    "save_period": track_instances.save_period,
                    "mem_bank": track_instances.mem_bank,
                },
            )
        mem_padding_mask, mem_bank, save_period = output
        track_instances.mem_padding_mask = mem_padding_mask
        track_instances.mem_bank = mem_bank
        track_instances.save_period = save_period

    return track_instances


def query_interact(models, track_instances):
    active_track_instances = track_instances[track_instances.obj_idxes >= 0]

    net = models["query_interact"]
    if not args.onnx:
        output = net.predict(
            [active_track_instances.query, active_track_instances.output_embedding]
        )
    else:
        output = net.run(
            None,
            {
                "query": active_track_instances.query,
                "output_embedding": active_track_instances.output_embedding,
            },
        )
    updated_query = output[0]
    active_track_instances.query = updated_query

    merged_track_instances = Instances.cat([empty_tracks(), active_track_instances])

    return merged_track_instances


def seg_head_forward(models, pts_feats):
    net = models["seg_head"]
    if not args.onnx:
        output = net.predict([pts_feats])
    else:
        output = net.run(
            None,
            {"bev_embed": pts_feats},
        )
    (memory, memory_mask, memory_pos, query, query_pos, hw_lvl, hw_lvl2) = output

    out = [
        memory,
        memory_mask,
        memory_pos,
        query,
        query_pos,
        [hw_lvl, hw_lvl2],
    ]
    return out


def motion_head_forward(models, bev_embed, outs_track: dict, outs_seg: dict):
    track_query = outs_track["track_query_embeddings"][None, None, ...]
    track_boxes: LiDARInstance3DBoxes = outs_track["track_bbox_results"]

    track_query = np.concatenate(
        [track_query, outs_track["sdc_embedding"][None, None, None, :]], axis=2
    )
    sdc_track_boxes = outs_track["sdc_track_bbox_results"]

    track_boxes[0][0].tensor = np.concatenate(
        [track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], axis=0
    )
    track_boxes[0][1] = np.concatenate(
        [track_boxes[0][1], sdc_track_boxes[0][1]], axis=0
    )
    track_boxes[0][2] = np.concatenate(
        [track_boxes[0][2], sdc_track_boxes[0][2]], axis=0
    )
    track_boxes[0][3] = np.concatenate(
        [track_boxes[0][3], sdc_track_boxes[0][3]], axis=0
    )
    memory, memory_mask, memory_pos, lane_query, lane_query_pos, hw_lvl = outs_seg

    net = models["motion_head"]
    if not args.onnx:
        output = net.predict(
            [
                bev_embed,
                track_query,
                lane_query,
                lane_query_pos,
                track_boxes[0][0].tensor,
                track_boxes[0][2],
            ]
        )
    else:
        output = net.run(
            None,
            {
                "bev_embed": bev_embed,
                "track_query": track_query,
                "lane_query": lane_query,
                "lane_query_pos": lane_query_pos,
                "bboxes": track_boxes[0][0].tensor,
                "bbox_labels": track_boxes[0][2],
            },
        )
    (
        outputs_traj_scores,
        outputs_trajs,
        valid_traj_masks,
        inter_states,
        track_query,
        track_query_pos,
    ) = output

    outs_motion = {
        "all_traj_scores": outputs_traj_scores,
        "all_traj_preds": outputs_trajs,
        "valid_traj_masks": valid_traj_masks,
        "traj_query": inter_states,
        "track_query": track_query,
        "track_query_pos": track_query_pos,
    }

    # get_trajs
    traj_results = get_trajs(outs_motion, track_boxes)
    bboxes, scores, labels, bbox_index, mask = track_boxes[0]
    outs_motion["track_scores"] = scores[None, :]
    labels[-1] = 0

    def filter_vehicle_query(outs_motion, labels, vehicle_id_list):
        if len(labels) < 1:  # No other obj query except sdc query.
            return None

        # select vehicle query according to vehicle_id_list
        vehicle_mask = np.zeros_like(labels)
        for veh_id in vehicle_id_list:
            vehicle_mask |= labels == veh_id
        outs_motion["traj_query"] = outs_motion["traj_query"][:, :, vehicle_mask > 0]
        outs_motion["track_query"] = outs_motion["track_query"][:, vehicle_mask > 0]
        outs_motion["track_query_pos"] = outs_motion["track_query_pos"][
            :, vehicle_mask > 0
        ]
        outs_motion["track_scores"] = outs_motion["track_scores"][:, vehicle_mask > 0]
        return outs_motion

    # Define vehicle_id_list as a default list
    vehicle_id_list = [
        0,
        1,
        2,
        3,
        4,
        6,
        7,
    ]  # car, truck, bus, trailer, construction_vehicle, motorcycle, bicycle, pedestrian, traffic_cone
    outs_motion = filter_vehicle_query(outs_motion, labels, vehicle_id_list)

    # filter sdc query
    outs_motion["sdc_traj_query"] = outs_motion["traj_query"][:, :, -1]
    outs_motion["sdc_track_query"] = outs_motion["track_query"][:, -1]
    outs_motion["sdc_track_query_pos"] = outs_motion["track_query_pos"][:, -1]
    outs_motion["traj_query"] = outs_motion["traj_query"][:, :, :-1]
    outs_motion["track_query"] = outs_motion["track_query"][:, :-1]
    outs_motion["track_query_pos"] = outs_motion["track_query_pos"][:, :-1]
    outs_motion["track_scores"] = outs_motion["track_scores"][:, :-1]

    return traj_results, outs_motion


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
            # "type": "NuScenesE2EDataset",
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
            # "file_client_args": {"backend": "disk"},
            # "patch_size": [102.4, 102.4],
            # "canvas_size": (200, 200),
            # "bev_size": (200, 200),
            # "predict_steps": 12,
            # "past_steps": 4,
            # "fut_steps": 4,
            # "occ_n_future": 6,
            # "use_nonlinear_optimizer": True,
            # "eval_mod": ["det", "map", "track", "motion"],
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
        bev_embed = output["bev_embed"]
        output_classes = output["pred_logits"]
        output_coords = output["pred_boxes"]
        last_ref_pts = output["ref_pts"]
        query_feats = output["query_embeddings"]

        item = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            # "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            # "all_past_traj_preds": det_output["all_past_traj_preds"],
            # "bev_pos": bev_pos,
        }

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
        """ update track base """
        track_base.update(track_instances)

        filter_score_thresh = 0.35
        active_index = (track_instances.obj_idxes >= 0) & (
            track_instances.scores >= filter_score_thresh
        )  # filter out sleep objects
        # select_active_track_query
        result_dict = track_instances2results(
            track_instances[active_index], with_mask=True
        )
        result_dict["track_query_embeddings"] = track_instances.output_embedding[
            active_index
        ][result_dict["bbox_index"]][result_dict["mask"]]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[
            active_index
        ][result_dict["bbox_index"]][result_dict["mask"]]
        item.update(result_dict)
        # select_sdc_track_query
        sdc_instance = track_instances[track_instances.obj_idxes == -2]
        result_dict = track_instances2results(sdc_instance, with_mask=False)
        item.update(
            dict(
                sdc_boxes_3d=result_dict["boxes_3d"],
                sdc_scores_3d=result_dict["scores_3d"],
                sdc_track_scores=result_dict["track_scores"],
                sdc_track_bbox_results=result_dict["track_bbox_results"],
                sdc_embedding=sdc_instance.output_embedding[0],
            )
        )

        """ update with memory_bank """
        track_instances = memory_bank(models, track_instances)

        """  Update track instances using matcher """
        track_instances_fordet = track_instances
        track_instances = query_interact(models, track_instances)

        result_dict = dict(
            bev_embed=bev_embed,
            # bev_pos=item["bev_pos"],
            track_query_embeddings=item["track_query_embeddings"],
            track_bbox_results=item["track_bbox_results"],
            boxes_3d=item["boxes_3d"],
            scores_3d=item["scores_3d"],
            labels_3d=item["labels_3d"],
            track_scores=item["track_scores"],
            track_ids=item["track_ids"],
            sdc_boxes_3d=item["sdc_boxes_3d"],
            sdc_scores_3d=item["sdc_scores_3d"],
            sdc_track_scores=item["sdc_track_scores"],
            sdc_track_bbox_results=item["sdc_track_bbox_results"],
            sdc_embedding=item["sdc_embedding"],
        )
        result_track = det_instances2results(track_instances_fordet, result_dict)

        # seg_head
        result_seg = seg_head_forward(models, bev_embed)
        # motion_head
        result_motion, outs_motion = motion_head_forward(
            models, bev_embed, outs_track=result_track, outs_seg=result_seg
        )

        result = dict()
        result["token"] = img_metas["sample_idx"]
        # result["occ"] = outs_occ
        # result["planning"] = dict(
        #     planning_gt=planning_gt,
        #     result_planning=result_planning,
        # )
        result.update(result_track)
        result.update(
            result_motion[0]
        )  # 'traj_0', 'traj_scores_0', 'traj_1', 'traj_scores_1', 'traj', 'traj_scores'
        # result.update(result_seg)  # ret_iou

        ### End of forward_test ###

        bbox_results.append(result)
        # occ_results_computed = occ_results
        # planning_results_computed = occ_results
        # mask_results = mask_results

        ### End of custom_multi_gpu_test ###

    vis = Visualizer(
        version="v1.0-mini",
        bbox_results=bbox_results,
        dataroot="data/nuscenes",
    )
    for i in range(len(vis.nusc.sample)):
        sample_token = vis.nusc.sample[i]["token"]

        vis.visualize_bev()
        vis.visualize_cam(sample_token, os.path.join("vis_output", str(i).zfill(3)))

    to_video("vis_output", "output_video.avi", fps=4, downsample=2)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    # check_and_download_models(WEIGHT_XXX_PATH, MODEL_XXX_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        bev_encoder = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
        track_head = ailia.Net(
            MODEL_TRACK_HEAD_PATH, WEIGHT_TRACK_HEAD_PATH, env_id=env_id
        )
        memory_bank = ailia.Net(
            MODEL_MEMORY_BANK_PATH, WEIGHT_MEMORY_BANK_PATH, env_id=env_id
        )
        memory_bank_update = ailia.Net(
            MODEL_MEMORY_BANK_UPD_PATH, WEIGHT_MEMORY_BANK_UPD_PATH, env_id=env_id
        )
        query_interact = ailia.Net(
            MODEL_QUERY_INTERACTION_PATH, WEIGHT_QUERY_INTERACTION_PATH, env_id=env_id
        )
        seg_head = ailia.Net(MODEL_SEG_HEAD_PATH, WEIGHT_SEG_HEAD_PATH, env_id=env_id)
        motion_head = ailia.Net(
            MODEL_MOTION_HEAD_PATH, WEIGHT_MOTION_HEAD_PATH, env_id=env_id
        )
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        bev_encoder = onnxruntime.InferenceSession(WEIGHT_PATH, providers=providers)
        track_head = onnxruntime.InferenceSession(
            WEIGHT_TRACK_HEAD_PATH, providers=providers
        )
        memory_bank = onnxruntime.InferenceSession(
            WEIGHT_MEMORY_BANK_PATH, providers=providers
        )
        memory_bank_update = onnxruntime.InferenceSession(
            WEIGHT_MEMORY_BANK_UPD_PATH, providers=providers
        )
        query_interact = onnxruntime.InferenceSession(
            WEIGHT_QUERY_INTERACTION_PATH, providers=providers
        )
        seg_head = onnxruntime.InferenceSession(
            WEIGHT_SEG_HEAD_PATH, providers=providers
        )
        motion_head = onnxruntime.InferenceSession(
            WEIGHT_MOTION_HEAD_PATH, providers=providers
        )

    models = dict(
        bev_encoder=bev_encoder,
        track_head=track_head,
        memory_bank=memory_bank,
        memory_bank_update=memory_bank_update,
        query_interact=query_interact,
        seg_head=seg_head,
        motion_head=motion_head,
    )
    recognize_from_image(models)


if __name__ == "__main__":
    main()

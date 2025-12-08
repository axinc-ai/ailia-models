import os
import pickle

import cv2
import numpy as np
import tqdm
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.prediction import PredictHelper
from nuscenes.utils import splits
from torch.utils.data import Dataset


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if nusc.dataroot in data_path:
        data_path = data_path.split(f"{nusc.dataroot}")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get("scene", sample["scene_token"])["name"]
    sample_timestamp = sample["timestamp"]
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, "pose")
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop("utime")  # useless
    pos = last_pose.pop("pos")
    rotation = last_pose.pop("orientation")
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus)


def get_future_traj_info(nusc, sample, predict_steps=16):
    sample_token = sample["token"]
    ann_tokens = np.array(sample["anns"])
    sd_rec = nusc.get("sample", sample_token)
    fut_traj_all = []
    fut_traj_valid_mask_all = []
    _, boxes, _ = nusc.get_sample_data(
        sd_rec["data"]["LIDAR_TOP"], selected_anntokens=ann_tokens
    )
    predict_helper = PredictHelper(nusc)
    for i, ann_token in enumerate(ann_tokens):
        box = boxes[i]
        instance_token = nusc.get("sample_annotation", ann_token)["instance_token"]
        fut_traj_local = predict_helper.get_future_for_agent(
            instance_token,
            sample_token,
            seconds=predict_steps // 2,
            in_agent_frame=True,
        )

        fut_traj = np.zeros((predict_steps, 2))
        fut_traj_valid_mask = np.zeros((predict_steps, 2))
        if fut_traj_local.shape[0] > 0:
            fut_traj_scence_centric = fut_traj_local
            fut_traj[: fut_traj_scence_centric.shape[0], :] = fut_traj_scence_centric
            fut_traj_valid_mask[: fut_traj_scence_centric.shape[0], :] = 1
        fut_traj_all.append(fut_traj)
        fut_traj_valid_mask_all.append(fut_traj_valid_mask)
    if len(ann_tokens) > 0:
        fut_traj_all = np.stack(fut_traj_all, axis=0)
        fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)
    else:
        fut_traj_all = np.zeros((0, predict_steps, 2))
        fut_traj_valid_mask_all = np.zeros((0, predict_steps, 2))
    return fut_traj_all, fut_traj_valid_mask_all


class NuScenesDataset(Dataset):
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }

    def __init__(
        self,
        ann_file=None,
        data_root=None,
        classes=None,
    ):
        if data_root is None:
            self.data_root = "./"
        elif data_root.endswith("/"):
            self.data_root = data_root
        else:
            self.data_root = data_root + "/"
        self.CLASSES = classes

        if ann_file is None:
            self.data_infos = self.create_nuscenes_infos(self.data_root)
        else:
            ann_file = os.path.join(self.data_root, "nuscenes_infos_val.pkl")
            self.data_infos = self.load_annotations(ann_file)

    def __len__(self):
        return len(self.data_infos)

    def create_nuscenes_infos(self, data_root, max_sweeps=10):
        version = "v1.0-trainval"
        # val_scenes = ["scene-0103", "scene-0916"]
        val_scenes = splits.val

        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        nusc_can_bus = NuScenesCanBus(dataroot=data_root)

        available_scenes = nusc.scene
        available_scene_names = [s["name"] for s in available_scenes]
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        val_scenes = set(
            [
                available_scenes[available_scene_names.index(s)]["token"]
                for s in val_scenes
            ]
        )
        samples = [x for x in nusc.sample if x["scene_token"] in val_scenes]

        val_nusc_infos = []
        frame_idx = 0
        for sample in tqdm.tqdm(samples):
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

            can_bus = get_can_bus_info(nusc, nusc_can_bus, sample)
            info = {
                "lidar_path": lidar_path,
                "token": sample["token"],
                "prev": sample["prev"],
                "next": sample["next"],
                "can_bus": can_bus,
                "frame_idx": frame_idx,  # temporal related info
                "sweeps": [],
                "cams": dict(),
                "scene_token": sample["scene_token"],  # temporal related info
                "lidar2ego_translation": cs_record["translation"],
                "lidar2ego_rotation": cs_record["rotation"],
                "ego2global_translation": pose_record["translation"],
                "ego2global_rotation": pose_record["rotation"],
                "timestamp": sample["timestamp"],
            }

            if sample["next"] == "":
                frame_idx = 0
            else:
                frame_idx += 1

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info.update(cam_intrinsic=cam_intrinsic)
                info["cams"].update({cam: cam_info})

            # obtain sweeps for a single key-frame
            sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            sweeps = []
            while len(sweeps) < max_sweeps:
                if not sd_rec["prev"] == "":
                    sweep = obtain_sensor2top(
                        nusc,
                        sd_rec["prev"],
                        l2e_t,
                        l2e_r_mat,
                        e2g_t,
                        e2g_r_mat,
                        "lidar",
                    )
                    sweeps.append(sweep)
                    sd_rec = nusc.get("sample_data", sd_rec["prev"])
                else:
                    break
            info["sweeps"] = sweeps
            # obtain annotation
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            valid_flag = np.array(
                [
                    (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                    for anno in annotations
                ],
                dtype=bool,
            ).reshape(-1)
            instance_inds = [
                nusc.getind("instance", ann["instance_token"]) for ann in annotations
            ]
            future_traj_all, future_traj_valid_mask_all = get_future_traj_info(
                nusc, sample
            )
            instance_tokens = [
                ann["instance_token"] for ann in annotations
            ]  # dtype('<U[length_of_str]')

            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # instance_inds = [nusc.getind('instance', ann['instance_token']) for ann in annotations]
            # TODO(box3d): convert gt_boxes to mmdet3d 1.0.0rc6 LiDARInstance3DBoxes format. [DONE]
            gt_boxes = np.concatenate([locs, dims, rots], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag
            info["gt_inds"] = np.array(instance_inds)
            info["gt_ins_tokens"] = np.array(instance_tokens)
            info["fut_traj"] = future_traj_all
            info["fut_traj_valid_mask"] = future_traj_valid_mask_all

            # add visibility_tokens
            visibility_tokens = [int(anno["visibility_token"]) for anno in annotations]
            info["visibility_tokens"] = np.array(visibility_tokens)

            val_nusc_infos.append(info)

        return val_nusc_infos

    def load_annotations(self, ann_file):
        """Load annotations from ann_file."""
        with open(ann_file, "rb") as f:
            value_buf = f.read()
            data = pickle.loads(value_buf)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]

        return data_infos

    def __getitem__(self, idx):
        input_dict = self.get_data_info(idx)
        data_dict = self.pipeline(input_dict)

        meta_keys = [
            # "filename",
            # "ori_shape",
            "sample_idx",
            "img_shape",
            "lidar2img",
            # "depth2img",
            # "cam2img",
            # "pad_shape",
            # "scale_factor",
            # "flip",
            # "pcd_horizontal_flip",
            # "pcd_vertical_flip",
            # "box_mode_3d",
            # "box_type_3d",
            # "img_norm_cfg",
            # "pcd_trans",
            # "prev_idx",
            # "next_idx",
            # "pcd_scale_factor",
            # "pcd_rotation",
            # "pts_filename",
            # "transformation_3d_flow",
            "scene_token",
            "can_bus",
        ]
        img_metas = {}
        for key in meta_keys:
            if key in data_dict:
                img_metas[key] = data_dict[key]
        img_metas["lidar2img"] = np.array(img_metas["lidar2img"])
        img_metas["img_shape"] = np.array(img_metas["img_shape"])
        data_dict["img_metas"] = img_metas

        return data_dict

    def get_data_info(self, index):
        """Get data info according to the given index."""
        info = self.data_infos[index]

        input_dict = dict(
            sample_idx=info["token"],
            # pts_filename=info["lidar_path"],
            # sweeps=info["sweeps"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            # prev_idx=info["prev"],
            # next_idx=info["next"],
            scene_token=info["scene_token"],
            can_bus=info["can_bus"],
            # frame_idx=info["frame_idx"],
            timestamp=info["timestamp"] / 1e6,
        )

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        input_dict.update(
            dict(l2g_r_mat=l2g_r_mat.astype(np.float32), l2g_t=l2g_t.astype(np.float32))
        )

        # use_camera
        image_paths = []
        lidar2img_rts = []
        for cam_type, cam_info in info["cams"].items():
            image_paths.append(cam_info["data_path"])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
            lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info["cam_intrinsic"]
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts.append(lidar2img_rt)
        input_dict["img_filename"] = image_paths
        input_dict["lidar2img"] = lidar2img_rts

        rotation = Quaternion(input_dict["ego2global_rotation"])
        translation = input_dict["ego2global_translation"]
        can_bus = input_dict["can_bus"]
        can_bus[:3] = translation
        # NOTE(lty): fix can_bus format, in https://github.com/OpenDriveLab/UniAD/pull/214
        can_bus[3:7] = rotation.elements
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def pipeline(self, data):
        results = data

        ### LoadMultiViewImageFromFilesInCeph

        filename = results["img_filename"]
        images_multiView = []
        for img_path in filename:
            img_path = os.path.join(self.data_root, img_path)
            img = cv2.imread(img_path)
            images_multiView.append(img)
        img = np.stack(
            images_multiView,
            axis=-1,
        )
        img = img.astype(np.float32)

        results["filename"] = filename
        results["img"] = ori_img = [img[..., i] for i in range(img.shape[-1])]

        ### NormalizeMultiviewImage

        def imnormalize(img, mean, std):
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            img = (img - mean) / std
            return img

        mean = np.array([103.53, 116.28, 123.675])
        std = np.array([1, 1, 1])
        results["img"] = norm_img = [imnormalize(img, mean, std) for img in ori_img]

        ### PadMultiViewImage

        def impad_to_multiple(img, divisor, pad_val=0):
            pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
            pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
            # shape = (pad_h, pad_w)
            width = max(pad_w - img.shape[1], 0)
            height = max(pad_h - img.shape[0], 0)
            padding = (0, 0, width, height)

            img = cv2.copyMakeBorder(
                img,
                padding[1],
                padding[3],
                padding[0],
                padding[2],
                cv2.BORDER_CONSTANT,
                value=pad_val,
            )
            return img

        size_divisor = 32
        results["img"] = padded_img = [
            impad_to_multiple(img, size_divisor) for img in norm_img
        ]
        results["img_shape"] = [img.shape for img in padded_img]

        return results

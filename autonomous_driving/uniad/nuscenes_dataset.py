import os
import pickle

import cv2
import numpy as np
from torch.utils.data import Dataset
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


class NuScenesDataset(Dataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
    ):
        self.data_root = data_root

        self.data_infos = self.load_annotations(ann_file)
        self.nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=True
        )

    def __len__(self):
        return len(self.data_infos)

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
        example = self.pipeline(input_dict)
        data_dict = {}
        for key, value in example.items():
            if "l2g" in key:
                data_dict[key] = to_tensor(value[0])
            else:
                data_dict[key] = value
        return data_dict

    def get_data_info(self, index):
        """Get data info according to the given index."""
        info = self.data_infos[index]

        location = self.nusc.get(
            "log", self.nusc.get("scene", info["scene_token"])["log_token"]
        )["location"]

        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            prev_idx=info["prev"],
            next_idx=info["next"],
            scene_token=info["scene_token"],
            can_bus=info["can_bus"],
            frame_idx=info["frame_idx"],
            timestamp=info["timestamp"] / 1e6,
        )

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
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["scale_factor"] = 1.0

        ### NormalizeMultiviewImage

        def _imnormalize(img, mean, std):
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            img = (img - mean) / std
            return img

        mean = np.array([103.53, 116.28, 123.675])
        std = np.array([1, 1, 1])
        results["img"] = [_imnormalize(img, mean, std) for img in results["img"]]

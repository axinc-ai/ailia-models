import os
import os.path

import h5py
import numpy as np

import dataloaders.transforms as transforms


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f["rgb"])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f["depth"])
    return rgb, depth


class MyDataloader(object):
    modality_names = ["rgb"]

    def is_image_file(self, filename):
        IMG_EXTENSIONS = [".h5"]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, split, modality="rgb", loader=h5_loader):
        classes, class_to_idx = self.find_classes(root)
        imgs = self.make_dataset(root, class_to_idx)
        assert len(imgs) > 0, "Found 0 images in subfolders of: " + root + "\n"
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if split == "train":
            self.transform = self.train_transform
        elif split == "holdout":
            self.transform = self.val_transform
        elif split == "val":
            self.transform = self.val_transform
        else:
            raise (
                RuntimeError(
                    "Invalid dataset split: " + split + "\n"
                    "Supported dataset splits are: train, val"
                )
            )
        self.loader = loader

        assert modality in self.modality_names, (
            "Invalid modality split: "
            + modality
            + "\n"
            + "Supported dataset splits are: "
            + "".join(self.modality_names)
        )
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)

        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise (RuntimeError("transform not defined"))

        if self.modality == "rgb":
            input_np = rgb_np

        input_tensor = input_np.transpose((2, 0, 1)).copy()
        while input_tensor.ndim < 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        depth_tensor = depth_np.copy()
        depth_tensor = np.expand_dims(depth_tensor, 0)

        return np.expand_dims(input_tensor, 0), np.expand_dims(depth_tensor, 0)

    def __len__(self):
        return len(self.imgs)

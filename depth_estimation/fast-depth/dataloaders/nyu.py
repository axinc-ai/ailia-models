import numpy as np

import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640  # raw image size


class NYUDataset(MyDataloader):
    def __init__(self, root, split, modality="rgb"):
        self.split = split
        super(NYUDataset, self).__init__(root, split, modality)
        self.output_size = (224, 224)

    def is_image_file(self, filename):
        # IMG_EXTENSIONS = ['.h5']
        if self.split == "train":
            return (
                filename.endswith(".h5")
                and "00001.h5" not in filename
                and "00201.h5" not in filename
            )
        elif self.split == "holdout":
            return "00001.h5" in filename or "00201.h5" in filename
        elif self.split == "val":
            return filename.endswith(".h5")
        else:
            raise (
                RuntimeError(
                    "Invalid dataset split: " + self.split + "\n"
                    "Supported dataset splits are: train, val"
                )
            )

    def val_transform(self, rgb, depth_np):
        transform = transforms.Compose(
            [
                transforms.Resize((int(iwidth * (250.0 / iheight)), 250)),
                transforms.CenterCrop((228, 304)),
                transforms.Resize(self.output_size),
            ]
        )
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype="float") / 255
        if depth_np is not None:
            depth_np = transform(depth_np)

        return rgb_np, depth_np

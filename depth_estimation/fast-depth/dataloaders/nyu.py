from dataloaders.dataloader import MyDataloader
from dataloaders.utils import val_transform


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
                    f"Invalid dataset split: {self.split}\n"
                    "Supported dataset splits are: train, val"
                )
            )

    def val_transform(self, rgb, depth_np):
        return val_transform(rgb, depth_np, self.output_size)

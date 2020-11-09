from pathlib import Path
import pickle

from PIL import Image
import cv2
import numpy as np


class DataReader:
    image_dir_name = "images"
    seg_dir_name = "segs"
    landmark_dir_name = "landmarks"
    makeup = "makeup.txt"
    non_makeup = "non-makeup.txt"

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir.joinpath(self.image_dir_name)
        self.seg_dir = self.data_dir.joinpath(self.seg_dir_name)
        self.lms_dir = self.data_dir.joinpath(self.landmark_dir_name)
        self.makeup_names = [name.strip() for name in self.data_dir.joinpath(self.makeup).open("rt")]
        self.non_makeup_names = [name.strip() for name in self.data_dir.joinpath(self.non_makeup).open("rt")]

        self.random = None

    def read_file(self, name):
        image = Image.open(
            self.image_dir.joinpath(name).as_posix()
        ).convert("RGB")
        seg = np.asarray(
            Image.open(
                self.seg_dir.joinpath(name).as_posix()
            )
        )
        lm = pickle.load(self.lms_dir.joinpath(name).open("rb"))

        return image, seg, lm

    def __getitem__(self, index):
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        if isinstance(index, tuple):
            assert len(index) == 2
            index_non_makeup = index[1]
            index = index[0]
        else:
            assert isinstance(index, int)
            index_non_makeup = index

        return self.read_file(self.non_makeup_names[index_non_makeup]),\
            self.read_file(self.makeup_names[index])

    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def pick(self):
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        a_index = self.random.randint(0, len(self.makeup_names))
        another_index = self.random.randint(0, len(self.non_makeup_names))
        return self[a_index, another_index]

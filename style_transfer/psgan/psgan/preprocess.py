import os.path as osp
pwd = osp.split(osp.realpath(__file__))[0]
import sys
sys.path.append(pwd + '/..')

import cv2
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

import faceutils as futils

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])


def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


def to_var(x, requires_grad=True):
    if requires_grad:
        return Variable(x).float()
    else:
        return Variable(x, requires_grad=requires_grad).float()


def copy_area(tar, src, lms):
    rect = [int(min(lms[:, 1])) - PreProcess.eye_margin, 
            int(min(lms[:, 0])) - PreProcess.eye_margin, 
            int(max(lms[:, 1])) + PreProcess.eye_margin + 1, 
            int(max(lms[:, 0])) + PreProcess.eye_margin + 1]
    tar[:, :, rect[1]:rect[3], rect[0]:rect[2]] = \
        src[:, :, rect[1]:rect[3], rect[0]:rect[2]]
    src[:, :, rect[1]:rect[3], rect[0]:rect[2]] = 0


class PreProcess:
    eye_margin = 16
    diff_size = (64, 64)

    def __init__(self, config, device="cpu", need_parser=True):
        self.device = device
        self.img_size    = config.DATA.IMG_SIZE

        xs, ys = np.meshgrid(
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            ),
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            )
        )
        xs = xs[None].repeat(config.PREPROCESS.LANDMARK_POINTS, axis=0)
        ys = ys[None].repeat(config.PREPROCESS.LANDMARK_POINTS, axis=0)
        fix = np.concatenate([ys, xs], axis=0)
        self.fix = torch.Tensor(fix).to(self.device)
        if need_parser:
            self.face_parse = futils.mask.FaceParser(device=device)
        self.up_ratio    = config.PREPROCESS.UP_RATIO
        self.down_ratio  = config.PREPROCESS.DOWN_RATIO
        self.width_ratio = config.PREPROCESS.WIDTH_RATIO
        self.lip_class   = config.PREPROCESS.LIP_CLASS
        self.face_class  = config.PREPROCESS.FACE_CLASS

    def relative2absolute(self, lms):
        return lms * self.img_size

    def process(self, mask, lms, device="cpu"):
        diff = to_var(
            (self.fix.double() - torch.tensor(lms.transpose((1, 0)
                ).reshape(-1, 1, 1)).to(self.device)
            ).unsqueeze(0), requires_grad=False).to(self.device)

        lms_eye_left = lms[42:48]
        lms_eye_right = lms[36:42]
        lms = lms.transpose((1, 0)).reshape(-1, 1, 1)   # transpose to (y-x)
        # lms = np.tile(lms, (1, 256, 256))  # (136, h, w)
        diff = to_var((self.fix.double() - torch.tensor(lms).to(self.device)).unsqueeze(0), requires_grad=False).to(self.device)

        mask_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()

        mask_eyes = torch.zeros_like(mask, device=device)
        copy_area(mask_eyes, mask_face, lms_eye_left)
        copy_area(mask_eyes, mask_face, lms_eye_right)
        mask_eyes = to_var(mask_eyes, requires_grad=False).to(device)

        mask_list = [mask_lip, mask_face, mask_eyes]
        mask_aug = torch.cat(mask_list, 0)      # (3, 1, h, w)
        mask_re = F.interpolate(mask_aug, size=self.diff_size).repeat(1, diff.shape[1], 1, 1)  # (3, 136, 64, 64)
        diff_re = F.interpolate(diff, size=self.diff_size).repeat(3, 1, 1, 1)  # (3, 136, 64, 64)
        diff_re = diff_re * mask_re             # (3, 136, 32, 32)
        norm = torch.norm(diff_re, dim=1, keepdim=True).repeat(1, diff_re.shape[1], 1, 1)
        norm = torch.where(norm == 0, torch.tensor(1e10, device=device), norm)
        diff_re /= norm

        return mask_aug, diff_re

    def __call__(self, image: Image):
        face = futils.dlib.detect(image)

        if not face:
            return None, None, None

        face_on_image = face[0]

        image, face, crop_face = futils.dlib.crop(
            image, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        np_image = np.array(image)
        mask = self.face_parse.parse(cv2.resize(np_image, (512, 512)))
        # obtain face parsing result
        # image = image.resize((512, 512), Image.ANTIALIAS)
        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (self.img_size, self.img_size),
            mode="nearest")
        mask = mask.type(torch.uint8)
        mask = to_var(mask, requires_grad=False).to(self.device)

        # detect landmark
        lms = futils.dlib.landmarks(image, face) * self.img_size / image.width
        lms = lms.round()

        mask, diff = self.process(mask, lms, device=self.device)
        image = image.resize((self.img_size, self.img_size), Image.ANTIALIAS)
        image = transform(image)
        real = to_var(image.unsqueeze(0))
        return [real, mask, diff], face_on_image, crop_face

import os.path as osp
pwd = osp.split(osp.realpath(__file__))[0]
import sys
sys.path.append(pwd + '/..')

import ailia
import cv2
import dlib
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

import faceutils as futils
sys.path.append("../../util")
from image_utils import load_image  # noqa: E402

FACE_ALIGNMENT_IMAGE_HEIGHT = 256
FACE_ALIGNMENT_IMAGE_WIDTH = 256

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


class FaceAlignment:
    def __init__(self, use_onnx, face_alignment_model, face_alignment_weight):
        self.face_alignment_model = face_alignment_model
        self.face_alignment_weight = face_alignment_weight
        self.use_onnx = use_onnx
        # initialize net
        env_id = ailia.get_gpu_environment_id()
        print(f"env_id (face alignment): {env_id}")
        if not self.use_onnx:
            self.net = ailia.Net(self.face_alignment_model, self.face_alignment_weight, env_id=env_id)
        else:
            import onnxruntime
            self.net = onnxruntime.InferenceSession(self.face_alignment_weight)

    def predict(self, data):
        # return self.net.predict(data)
        if not self.use_onnx:
            return self.net.predict(data)
        else:
            inputs = {self.net.get_inputs()[0].name: data}
            return np.array(self.net.run(None, inputs))[0]


class PreProcess:
    eye_margin = 16
    diff_size = (64, 64)

    def __init__(self, config, device="cpu", args=None, face_parser_path=None, face_alignment_path=None, need_parser=True):
        self.device = device
        self.img_size = config.DATA.IMG_SIZE

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
            self.face_parse = futils.mask.FaceParser(device=device, args=args, face_parser_path=face_parser_path)
        self.up_ratio    = config.PREPROCESS.UP_RATIO
        self.down_ratio  = config.PREPROCESS.DOWN_RATIO
        self.width_ratio = config.PREPROCESS.WIDTH_RATIO
        self.lip_class   = config.PREPROCESS.LIP_CLASS
        self.face_class  = config.PREPROCESS.FACE_CLASS
        self.use_onnx = args.onnx
        self.use_dlib = args.use_dlib
        if not self.use_dlib:
            self.input = args.source_path
            self.face_alignment = FaceAlignment(self.use_onnx, face_alignment_path[0], face_alignment_path[1])

    def relative2absolute(self, lms):
        return lms * self.img_size

    def detect_landmark(self, image, face):
        if self.use_dlib:
            predictor = dlib.shape_predictor(
                pwd + '/../faceutils/dlibutils/shape_predictor_68_face_landmarks.dat')
            lms = futils.dlib.landmarks(predictor, image, face) * self.img_size / image.width
            lms = lms.round()
        else:
            # prepare input data
            data = load_image(
                self.input,
                (FACE_ALIGNMENT_IMAGE_HEIGHT, FACE_ALIGNMENT_IMAGE_WIDTH),
                normalize_type='255',
                gen_input_ailia=True
            ).astype(np.float32)
            preds_ailia = self.face_alignment.predict(data)
            print(preds_ailia.shape)#(4, 1, 68, 64, 64)
            pts, _ = _get_preds_from_hm(preds_ailia)
            lms = pts.reshape(68, 2) * 4

        return lms

    def process(self, mask, lms, device="cpu"):
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

        lms = self.detect_landmark(image, face)

        mask, diff = self.process(mask, lms, device=self.device)
        image = image.resize((self.img_size, self.img_size), Image.ANTIALIAS)
        image = transform(image)
        real = to_var(image.unsqueeze(0))
        return [real, mask, diff], face_on_image, crop_face


# Copied from face_recognition/face_alignment/face_alignment.py
def _get_preds_from_hm(hm):
    """
    Obtain (x,y) coordinates given a set of N heatmaps.
    ref: 1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Parameters
    ----------
    hm : np.array

    Returns
    -------
    preds:
    preds_orig:

    """
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    idx = np.argmax(
        hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3]), axis=2
    )
    idx += 1
    preds = idx.reshape(idx.shape[0], idx.shape[1], 1)
    preds = np.tile(preds, (1, 1, 2)).astype(np.float)
    preds[..., 0] = (preds[..., 0] - 1) % hm.shape[3] + 1
    preds[..., 1] = np.floor((preds[..., 1] - 1) / (hm.shape[2])) + 1

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]]).astype(np.float)
                preds[i, j] = preds[i, j] + (np.sign(diff) * 0.25)

    preds += -0.5
    preds_orig = np.zeros_like(preds)

    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds_orig[i, j] = _transform(
                preds[i, j],  # point
                np.array([IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2]),  # center
                (IMAGE_HEIGHT + IMAGE_WIDTH) // 2,  # FIXME not sure... # scale
                hm.shape[2],  # resolution
                True,
            )
    return preds, preds_orig


def _transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to
            perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct
            or the inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = scale  # NOTE: originally, scale * 200
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.dot(t, _pt))[0:2]
    return new_point.astype(np.int)

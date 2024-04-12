import os
import numpy as np
from PIL import Image

from .transforms import transform_logits, get_affine_transform
from .ootd_parsing_utils import *

class SimpleFolderDataset:
    def __init__(self, root, input_size=[512, 512], transform=None):
        self.root = root
        self.input_size = input_size
        self.transform = transform
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)
        self.is_pil_image = False
        if isinstance(root, Image.Image):
            self.file_list = [root]
            self.is_pil_image = True
        elif os.path.isfile(root):
            self.file_list = [os.path.basename(root)]
            self.root = os.path.dirname(root)
        else:
            self.file_list = os.listdir(self.root)

    def __len__(self):
        return len(self.file_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def _transform(self, pic, mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
    #def _transform(self, pic, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Converts a numpy.ndarray (H x W x C) in the range [0, 255]
        to a numpy.ndarray of shape (C x H x W) in the range [0.0, 1.0],
        and then normalize it according to `mean` and `std`.
        """
        if pic.ndim == 2:
            pic = pic[:, :, None]

        if pic.dtype == np.uint8:
            pic = pic / 255.

        pic = (pic - mean) / std
        pic = pic.transpose((2, 0, 1))
        return pic.astype(np.float32)

    def __getitem__(self, index):
        if self.is_pil_image:
            img = np.asarray(self.file_list[index])[:, :, [2, 1, 0]]
        else:
            img_name = self.file_list[index]
            img_path = os.path.join(self.root, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self._transform(input)
        meta = {
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return np.expand_dims(input, axis=0), meta

class Parsing:
    def __init__(self, models, is_onnx):
        self.models = models
        self.is_onnx = is_onnx

    def __call__(self, input_image):
        parsed_image, face_mask = self.inference(input_image)
        return parsed_image, face_mask

    def inference(self, input_dir):
        # load datasetloader
        dataset = SimpleFolderDataset(root=input_dir, input_size=[512, 512])

        for batch in dataset:
            image, meta = batch
            c = meta['center']#.numpy()[0]
            s = meta['scale']#.numpy()[0]
            w = meta['width']#.numpy()[0]
            h = meta['height']#.numpy()[0]

            if self.is_onnx:
                output = self.models.atr.run(None, {self.models.atr.get_inputs()[0].name: image})[0]
                upsample_output = self.models.upsample_atr.run(None, {self.models.upsample_atr.get_inputs()[0].name: output})[0]
            else:
                output = self.models.atr.run(image)[0]
                upsample_output = self.models.upsample_atr.run(output)[0]

            self.models.release_atr()

            upsample_output = upsample_output.squeeze()
            # upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
            upsample_output = upsample_output.transpose((1, 2, 0))  # CHW -> HWC
            logits_result = transform_logits(upsample_output, c, s, w, h, input_size=[512, 512])

            # delete irregular classes, e.g. pants/ skirts over clothes
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result = np.pad(parsing_result, pad_width=1, mode='constant', constant_values=0)

            # try holefilling the clothes part
            arm_mask = (parsing_result == 14).astype(np.float32) \
                    + (parsing_result == 15).astype(np.float32)
            upper_cloth_mask = (parsing_result == 4).astype(np.float32) + arm_mask
            img = np.where(upper_cloth_mask, 255, 0)
            dst = hole_fill(img.astype(np.uint8))
            parsing_result_filled = dst / 255 * 4
            parsing_result_woarm = np.where(parsing_result_filled == 4, parsing_result_filled, parsing_result)

            # add back arm and refined hole between arm and cloth
            refine_hole_mask = refine_hole(parsing_result_filled.astype(np.uint8), parsing_result.astype(np.uint8),
                                        arm_mask.astype(np.uint8))
            parsing_result = np.where(refine_hole_mask, parsing_result, parsing_result_woarm)
            # remove padding
            parsing_result = parsing_result[1:-1, 1:-1]

        dataset_lip = SimpleFolderDataset(root=input_dir, input_size=[473, 473])

        for batch in dataset_lip:
            image, meta = batch
            c = meta['center']#.numpy()[0]
            s = meta['scale']#.numpy()[0]
            w = meta['width']#.numpy()[0]
            h = meta['height']#.numpy()[0]

            if self.is_onnx:
                output_lip = self.models.lip.run(None, {self.models.lip.get_inputs()[0].name: image})[0]
                upsample_output_lip = self.models.upsample_lip.run(None, {self.models.upsample_lip.get_inputs()[0].name: output_lip})[0]
            else:
                output_lip = self.models.lip.run(image)[0]
                upsample_output_lip = self.models.upsample_lip.run(output_lip)[0]

            self.models.release_lip()

            upsample_output_lip = upsample_output_lip.squeeze()
            # upsample_output_lip = upsample_output_lip.permute(1, 2, 0)  # CHW -> HWC
            upsample_output_lip = upsample_output_lip.transpose((1, 2, 0))  # CHW -> HWC
            logits_result_lip = transform_logits(upsample_output_lip, c, s, w, h, input_size=[473, 473])
            parsing_result_lip = np.argmax(logits_result_lip, axis=2)

        # add neck parsing result
        neck_mask = np.logical_and(np.logical_not((parsing_result_lip == 13).astype(np.float32)), (parsing_result == 11).astype(np.float32))
        # filter out small part of neck
        neck_mask = refine_mask(neck_mask)
        # Image.fromarray(((neck_mask > 0) * 127.5 + 127.5).astype(np.uint8)).save("neck_mask.jpg")
        parsing_result = np.where(neck_mask, 18, parsing_result)
        palette = get_palette(19)
        # parsing_result_path = os.path.join('parsed.png')
        output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        output_img.putpalette(palette)
        # output_img.save(parsing_result_path)
        # face_mask = torch.from_numpy((parsing_result == 11).astype(np.float32))
        face_mask = (parsing_result == 11).astype(np.float32)

        return output_img, face_mask

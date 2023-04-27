import cv2
import numpy as np

from image_utils import normalize_image

from . import common


def inference_segmentor(net, img):
    size = 768

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = normalize_image(img, normalize_type='ImageNet')
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    if not common.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'img': img})
    result = output[0]

    return result


class UniformerDetector:
    def __init__(self, net):
        self.net = net

    def __call__(self, img):
        result = inference_segmentor(self.net, img)
        res_img = show_result_pyplot(self.net, img, result, get_palette('ade'), opacity=1)

        return res_img

    def map2img(self, detected_map):
        return detected_map[:, :, ::-1]  # RGB -> BGR

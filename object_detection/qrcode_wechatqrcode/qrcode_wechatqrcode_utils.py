# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv # needs to have cv.wechat_qrcode_WeChatQRCode, which requires compile from source with opencv_contrib/modules/wechat_qrcode

class WeChatQRCode:
    def __init__(self, detect_prototxt_path=None, detect_model_path=None, sr_prototxt_path=None, sr_model_path=None):
        if detect_prototxt_path == None:
            #QRCodeDetector
            pass
        else:
            self._model = cv.wechat_qrcode_WeChatQRCode(
                detect_prototxt_path,
                detect_model_path,
                sr_prototxt_path,
                sr_model_path
            )

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backend_id):
        # self._model.setPreferableBackend(backend_id)
        if backend_id != 0:
            raise NotImplementedError("Backend {} is not supported by cv.wechat_qrcode_WeChatQRCode()")

    def setTarget(self, target_id):
        # self._model.setPreferableTarget(target_id)
        if target_id != 0:
            raise NotImplementedError("Target {} is not supported by cv.wechat_qrcode_WeChatQRCode()")

    def infer(self, image):
        return self._model.detectAndDecode(image)

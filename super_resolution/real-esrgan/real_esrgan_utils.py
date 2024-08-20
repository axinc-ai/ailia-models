import cv2
import numpy as np
import os

class RealESRGAN():

    def __init__(self, model, scale=4):
        self.scale = scale
        self.model = model

    def enhance(self, img, outscale=3.5):
        h_input, w_input = img.shape[0:2]

        img = img.astype(np.float32)
        max_range = 255
        img = img / max_range
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        output_img = self.model.run(img)[0]

        output_img = np.squeeze(output_img)
        output_img = np.clip(output_img, 0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output

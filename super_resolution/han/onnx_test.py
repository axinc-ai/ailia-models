import onnx
import onnxruntime

onnx_model = onnx.load("han_BIX2.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("han_BIX2.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

import numpy as np
import cv2

IMAGE_PATH = '000002_LR.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 194    # net.get_input_shape()[3]
IMAGE_WIDTH = 194     # net.get_input_shape()[2]
OUTPUT_HEIGHT = 194*2  # net.get_output_shape()[3]
OUTPUT_WIDTH = 194*2   # net.get_output.shape()[2]

# Loading and processing image
image = cv2.imread(IMAGE_PATH, int(True))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0
image = image.transpose((2, 0, 1))  # channel first
#print(image.shape)
x = image[np.newaxis, :, :, :] # (batch_size, channel, h, w)
x = x.astype(np.float32)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: x}
ort_outs = ort_session.run(None, ort_inputs)

#print(ort_outs[0])
#print(ort_outs[0].shape)
        
output_img = ort_outs[0][0].transpose((1, 2, 0))
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
output_img = output_img * 255
#print(output_img.shape)
cv2.imwrite(SAVE_IMAGE_PATH, output_img)

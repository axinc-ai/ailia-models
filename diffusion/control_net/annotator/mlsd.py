import cv2
import numpy as np

from . import comm


def pred_lines(
        image, model,
        input_shape=[512, 512],
        score_thr=0.10,
        dist_thr=20.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate([
        cv2.resize(image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA),
        np.ones([input_shape[0], input_shape[1], 1])], axis=-1)

    resized_image = resized_image.transpose((2, 0, 1))
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    batch_image = (batch_image / 127.5) - 1.0

    if not comm.onnx:
        output = model.predict([batch_image])
    else:
        output = model.run(None, {'img': batch_image})
    pts, pts_score, vmap = output
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * np.array(segments_list)  # 256 > 512
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio

    return lines


class MLSDdetector:
    def __init__(self, net):
        self.net = net

    def __call__(self, img):
        value_threshold = 0.1
        distance_threshold = 0.1

        img_output = np.zeros_like(img)
        lines = pred_lines(img, self.net, [img.shape[0], img.shape[1]], value_threshold, distance_threshold)
        for line in lines:
            x_start, y_start, x_end, y_end = [int(val) for val in line]
            cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)

        return img_output[:, :, 0]

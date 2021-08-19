import numpy as np
import cv2

def pred_lines(image, net, input_shape=[512, 512], score_thr=0.10,
               dist_thr=10.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA),
                                    np.ones([input_shape[0], input_shape[1], 1])], axis=-1)
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')

    output = net.run(batch_image)

    pts = output[0][0]
    pts_score = output[1][0]
    vmap = output[2][0]

    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        center = center.astype('int32')
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    if len(segments_list)==0:
        return []

    lines = 2 * np.array(segments_list)  # 256 > 512
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio

    return lines

def pred_lines_onnx(image, sess, input_details, output_details, input_shape=[512, 512], score_thr=0.10,
               dist_thr=20.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA),
                                    np.ones([input_shape[0], input_shape[1], 1])], axis=-1)
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')

    input_name  = input_details[0].name
    output_name1 = output_details[0].name
    output_name2 = output_details[1].name
    output_name3 = output_details[2].name

    output = sess.run([output_name1, output_name2 ,output_name3], {input_name: batch_image})

    pts = output[0][0]
    pts_score = output[1][0]
    vmap = output[2][0]

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
import math
import cv2
import ailia

import numpy as np

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if True:#config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def compute(net,original_img,offset_x,offset_y,scale_x,scale_y):
    shape=net.get_input_shape()

    IMAGE_WIDTH = shape[3]
    IMAGE_HEIGHT = shape[2]

    src_img = cv2.resize(original_img,(IMAGE_WIDTH,IMAGE_HEIGHT))

    w=src_img.shape[1]
    h=src_img.shape[0]

    image_size = [IMAGE_WIDTH, IMAGE_HEIGHT]

    input_data = src_img

    center=np.array([w/2, h/2], dtype=np.float32)
    scale = np.array([1, 1], dtype=np.float32)

    #BGR format
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    input_data = (input_data/255.0 - mean) / std
    input_data = input_data[np.newaxis, :, :, :].transpose((0, 3, 1, 2))

    output = net.predict(input_data)
    
    preds, maxvals = get_final_preds(output, [center], [scale])

    k_list = []
    ailia_to_mpi=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-1,-1]
    ailia_to_coco=[0,14,15,16,17,2,5,3,6,4,8,11,7,9,12,10,13,1,-1]
    total_score = 0
    num_valid_points = 0
    id = 0
    angle_x = 0
    angle_y = 0
    angle_z = 0
    for j in range(ailia.POSE_KEYPOINT_CNT):
        i = ailia_to_mpi[j]
        z=0
        interpolated=0
        if j==ailia.POSE_KEYPOINT_BODY_CENTER:
            x=(k_list[ailia.POSE_KEYPOINT_SHOULDER_CENTER].x+k_list[ailia.POSE_KEYPOINT_HIP_LEFT].x+k_list[ailia.POSE_KEYPOINT_HIP_RIGHT].x)/3
            y=(k_list[ailia.POSE_KEYPOINT_SHOULDER_CENTER].y+k_list[ailia.POSE_KEYPOINT_HIP_LEFT].y+k_list[ailia.POSE_KEYPOINT_HIP_RIGHT].y)/3
            score=min(min(k_list[ailia.POSE_KEYPOINT_SHOULDER_CENTER].score,k_list[ailia.POSE_KEYPOINT_HIP_LEFT].score),k_list[ailia.POSE_KEYPOINT_HIP_RIGHT].score)
            interpolated=1
        elif j==ailia.POSE_KEYPOINT_SHOULDER_CENTER:
            x=(k_list[ailia.POSE_KEYPOINT_SHOULDER_LEFT].x+k_list[ailia.POSE_KEYPOINT_SHOULDER_RIGHT].x)/2
            y=(k_list[ailia.POSE_KEYPOINT_SHOULDER_LEFT].y+k_list[ailia.POSE_KEYPOINT_SHOULDER_RIGHT].y)/2
            score=min(k_list[ailia.POSE_KEYPOINT_SHOULDER_LEFT].score,k_list[ailia.POSE_KEYPOINT_SHOULDER_RIGHT].score)
            interpolated=1
        else:
            x=preds[0,i,0] / src_img.shape[1]
            y=preds[0,i,1] / src_img.shape[0]
            score=maxvals[0,i,0]

        num_valid_points=num_valid_points+1
        total_score=total_score+score

        k = ailia.PoseEstimatorKeypoint(
            x = x * scale_x + offset_x,
            y = y * scale_y + offset_y,
            z_local = z,
            score = score,
            interpolated = interpolated,
        )
        k_list.append(k)
    
    total_score=total_score/num_valid_points

    r = ailia.PoseEstimatorObjectPose(
        points = k_list,
        total_score = total_score,
        num_valid_points = num_valid_points,
        id = id,
        angle_x = angle_x,
        angle_y = angle_y,
        angle_z = angle_z
    )

    return r


def keep_aspect(top_left,bottom_right,pose_img,pose):
    py1 = max(0,top_left[1])
    py2 = min(pose_img.shape[0],bottom_right[1])
    px1 = max(0,top_left[0])
    px2 = min(pose_img.shape[1],bottom_right[0])

    shape = pose.get_input_shape()
    aspect = shape[2]/shape[3]
    ow = (px2-px1)
    w = (py2-py1)/aspect
    px1 = px1 - (w-ow)/2
    px2 = px1 + w

    px1 = int(px1)
    px2 = int(px2)
    py1 = int(py1)
    py2 = int(py2)

    py1 = max(0,py1)
    py2 = min(pose_img.shape[0],py2)
    px1 = max(0,px1)
    px2 = min(pose_img.shape[1],px2)
    return px1,py1,px2,py2

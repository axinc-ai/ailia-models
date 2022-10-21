import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lib.gen_kpts import gen_video_kpts as hrnet_pose


h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


def coco_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    keypoints_h36m[:, 7, 0] += 2 * (
                keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]) * 2 / 3

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]

    return keypoints_h36m, valid_frames


def h36m_coco_format(keypoints, scores):
    assert len(keypoints.shape) == 4 and len(scores.shape) == 3

    h36m_kpts = []
    h36m_scores = []
    valid_frames = []

    for i in range(keypoints.shape[0]):
        kpts = keypoints[i]
        score = scores[i]

        new_score = np.zeros_like(score, dtype=np.float32)

        if np.sum(kpts) != 0.:
            kpts, valid_frame = coco_h36m(kpts)
            h36m_kpts.append(kpts)
            valid_frames.append(valid_frame)

            new_score[:, h36m_coco_order] = score[:, coco_order]
            new_score[:, 0] = np.mean(score[:, [11, 12]], axis=1, dtype=np.float32)
            new_score[:, 8] = np.mean(score[:, [5, 6]], axis=1, dtype=np.float32)
            new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
            new_score[:, 10] = np.mean(score[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

            h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)

    return h36m_kpts


def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 5

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=5)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=5)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]

        ax.plot(x, y, z, lw=2)
        ax.scatter(x, y, z)

    RADIUS = 0.8

    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def get_pose2D(human_model, pose_model, img):
    keypoints, scores = hrnet_pose(img, human_model, pose_model, det_dim=416, num_peroson=1)
    if keypoints is not None:
        keypoints = h36m_coco_format(keypoints, scores)
        return keypoints
    else:
        return None


def get_pose3D(model, img, keypoints, fig=None):
    i = 0
    frames = 351
    pad = (frames - 1) // 2

    ## 3D
    img_size = img.shape

    ## input frames
    start = max(0, i - pad)
    end = min(i + pad, len(keypoints[0]) - 1)

    input_2D_no = keypoints[0][start:end + 1]

    left_pad, right_pad = 0, 0
    if input_2D_no.shape[0] != frames:
        if i < pad:
            left_pad = pad - i
        if i > len(keypoints[0]) - pad - 1:
            right_pad = i + pad - (len(keypoints[0]) - 1)

        input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[:, :, 0] *= -1
    input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
    input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

    input_2D = input_2D[np.newaxis, :, :, :, :]

    input_2D = input_2D.astype('float32')
    output_3D_non_flip = model.run(input_2D[:, 0])[0]
    output_3D_flip = model.run(input_2D[:, 1])[0]

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    output_3D[:, :, 0, :] = 0
    output_3D[:, :, 0, :] = 0
    post_out = output_3D[0, 0]

    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(post_out, R=rot, t=0)
    post_out[:, 2] -= np.min(post_out[:, 2])

    input_2D_no = input_2D_no[pad]

    ## 2D
    image = show2Dpose(input_2D_no, copy.deepcopy(img))
    #cv2.imwrite(str(('%04d' % i)) + '_2D.png', image)

    ## 3D
    if fig is None:
        fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05)
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose(post_out, ax)
    return image, ax


def detect(img, human_model, pose_model2d, pose_model3d, fig=None):
    keypoints = get_pose2D(human_model, pose_model2d, img)
    if keypoints is None:
        return img, None, False
    else:
        image_pose2d, fig_pose3d = get_pose3D(pose_model3d, img, keypoints, fig)
        return image_pose2d, fig_pose3d, True


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
    args = list(args)
    result = func(*args)
    return result


def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = np.cross(qvec, v, axis=len(q.shape) - 1)
    uuv = np.cross(qvec, uv, axis=len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))



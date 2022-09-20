import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lib.preprocess import h36m_coco_format
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from lib.camera import *

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


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)


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



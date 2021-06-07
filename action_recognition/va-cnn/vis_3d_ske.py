import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import webcamera_utils

# each joint is connected to some other joint:
connections = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
               12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
lines = list(zip(range(26), connections))
lines.remove((1, 0))


def draw_frames(ax, frame, ax_lim):
    x, y, z = [frame[:, i] for i in range(3)]

    # draw joints
    ax.scatter(x, z, y, c='r', marker='o', s=15)
    # draw connections between joints
    for i, j in lines:
        ax.plot([x[i], x[j]], [z[i], z[j]], [y[i], y[j]], c='g', linewidth=2)

    set_axes_equal(ax, ax_lim)


def get_axes_lim(joints):
    joints = joints.reshape(-1, 3)

    ax_min = [joints[:, i].min() for i in range(3)]
    ax_max = [joints[:, i].max() for i in range(3)]
    ax_mid = [(ax_max[i] + ax_min[i]) / 2.0 for i in range(3)]
    max_range = np.array([ax_max[i] - ax_min[i] for i in range(3)]).max() / 2.0

    ax_lim = [(ax_mid[i] - max_range, ax_mid[i] + max_range) for i in range(3)]

    return ax_lim


def set_axes_equal(ax, ax_lim):
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.grid(False)

    ax.set_xlim(ax_lim[0][0], ax_lim[0][1])
    ax.set_ylim(ax_lim[2][0], ax_lim[2][1])
    ax.set_zlim(ax_lim[1][0], ax_lim[1][1])

    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()

    ax_range = [abs(lim[1] - lim[0]) for lim in [x_lim, y_lim, z_lim]]
    radius = np.array(ax_range).max()
    ax_mid = [(lim[0] + lim[1]) * 0.5 for lim in [x_lim, y_lim, z_lim]]
    ax_lim = [(ax_mid[i] - radius, ax_mid[i] + radius) for i in range(3)]

    ax.set_xlim3d(ax_lim[0][0], ax_lim[0][1])
    ax.set_ylim3d(ax_lim[1][0], ax_lim[1][1])
    ax.set_zlim3d(ax_lim[2][0], ax_lim[2][1])


def draw_ske_data(ske_data, save_path):
    joints = purge_ske(ske_data)
    ax_lim = get_axes_lim(joints)
    joints = joints.reshape(-1, 25, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.tight_layout()
    ax.set_aspect('auto')
    ax.view_init(12, -81)

    # rendering to video
    num_frame = joints.shape[0]
    fps = 30
    writer = None
    for i in range(num_frame):
        print('{}/{}      '.format(i, num_frame), end='\r')
        draw_frames(ax, joints[i], ax_lim)
        fig.canvas.draw()

        im = np.array(fig.canvas.renderer.buffer_rgba())
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        ax.clear()

        if writer is None:
            f_h = int(im.shape[0])
            f_w = int(im.shape[1])
            writer = webcamera_utils.get_writer(save_path, f_h, f_w, fps=fps)
        writer.write(im)

    writer.release()


def purge_ske(ske_joint):
    zero_row = []
    for i in range(len(ske_joint)):
        if (ske_joint[i, :] == np.zeros((1, 150))).all():
            zero_row.append(i)
    ske_joint = np.delete(ske_joint, zero_row, axis=0)
    if (ske_joint[:, 0:75] == np.zeros((ske_joint.shape[0], 75))).all():
        ske_joint = np.delete(ske_joint, range(75), axis=1)
    elif (ske_joint[:, 75:150] == np.zeros((ske_joint.shape[0], 75))).all():
        ske_joint = np.delete(ske_joint, range(75, 150), axis=1)
    return ske_joint

import matplotlib

matplotlib.use('Agg')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import axes3d, Axes3D

__all__ = [
    'DataLoader',
    'coco_h36m',
    'get_joints_info',
    'normalize_screen_coordinates',
    'receptive_field',
    'camera_to_world',
    'render_animation',
]

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


class DataLoader(object):
    def __init__(
            self, poses_2d, pad=0, causal_shift=0,
            kps_left=None, kps_right=None):
        self.poses_2d = poses_2d
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.pad = pad
        self.causal_shift = causal_shift

    def next_epoch(self):
        for seq_2d in self.poses_2d:
            batch_2d = np.expand_dims(
                np.pad(seq_2d,
                       ((
                            self.pad + self.causal_shift,
                            self.pad - self.causal_shift
                        ),
                        (0, 0), (0, 0)),
                       'edge'),
                axis=0)

        batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
        batch_2d[1, :, :, 0] *= -1
        batch_2d[1, :, self.kps_left + self.kps_right] = \
            batch_2d[1, :, self.kps_right + self.kps_left]

        yield batch_2d.astype(np.float32)


class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = parents
        self._joints_left = joints_left
        self._joints_right = joints_right

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents


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

    # half body: the joint of ankle and knee equal to hip
    # keypoints_h36m[:, [2, 3]] = keypoints_h36m[:, [1, 1]]
    # keypoints_h36m[:, [5, 6]] = keypoints_h36m[:, [4, 4]]

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    return keypoints_h36m, valid_frames


def get_joints_info(num_joints):
    # Body+toe keypoints
    if num_joints == 19:
        joints_left = [5, 6, 7, 8, 13, 14, 15]
        joints_right = [1, 2, 3, 4, 16, 17, 18]
        h36m_skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 10, 16, 17],
            joints_left=[5, 6, 7, 8, 13, 14, 15],
            joints_right=[1, 2, 3, 4, 16, 17, 18])
    # Body keypoints
    else:
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        h36m_skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            joints_left=[4, 5, 6, 11, 12, 13],
            joints_right=[1, 2, 3, 14, 15, 16])

    keypoints_metadata = {
        'keypoints_symmetry': (joints_left, joints_right),
        'layout_name': 'Human3.6M',
        'num_joints': num_joints}

    return joints_left, joints_right, h36m_skeleton, keypoints_metadata


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def receptive_field(filter_widths):
    """
    Return the total receptive field of this model as # of frames.
    """
    pad = [filter_widths[0] // 2]
    next_dilation = filter_widths[0]
    for i in range(1, len(filter_widths)):
        pad.append((filter_widths[i] - 1) * next_dilation // 2)
        next_dilation *= filter_widths[i]

    frames = 0
    for f in pad:
        frames += f
    return 1 + 2 * frames


def qort(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    import torch
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    import torch
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        result = result.numpy()
        return result
    else:
        return result


def camera_to_world(X, R, t):
    return wrap(qort, np.tile(R, (*X.shape[:-1], 1)), X) + t


elbow_knee_v1 = [5, 15]
elbow_knee_v2 = [2, 12]
wrist_ankle_v1 = [6, 16]
wrist_ankle_v2 = [3, 13]
hip_shoulder = [1, 4, 11, 14]
spine_head = [7, 9]
thorax_neck = [8, 10]


def color_edge(joint_num):
    if joint_num in elbow_knee_v1:
        color = 'peru'
    elif joint_num in elbow_knee_v2:
        color = 'indianred'
    elif joint_num in wrist_ankle_v1:
        color = 'coral'
    elif joint_num in wrist_ankle_v2:
        color = 'brown'
    elif joint_num in hip_shoulder:
        color = 'tan'
    elif joint_num in spine_head:
        color = 'olive'
    else:
        color = 'purple'
    return color


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(
        keypoints, keypoints_metadata, poses, skeleton, fps, bitrate,
        azim, output,
        capture, viewport,
        downsample=1, size=6):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.5
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        # ax.set_title(title)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Load video using ffmpeg
    all_frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)

    effective_length = min(keypoints.shape[0], len(all_frames))
    all_frames = all_frames[:effective_length]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    limit = len(all_frames)
    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = color_edge(j)
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], linewidth=3, zdir='z', c=col))

            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')

            points.set_offsets(keypoints[i])

        print('{}/{}      '.format(i, limit), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')

    plt.close()

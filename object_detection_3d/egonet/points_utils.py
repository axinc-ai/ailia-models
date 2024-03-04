import numpy as np
import matplotlib.pyplot as plt


def set_3d_axe_limits(ax, points=None, center=None, radius=None, ratio=1.2):
    """
    Set 3d axe limits to simulate set_aspect('equal').
    Matplotlib has not yet provided implementation of set_aspect('equal')
    for 3d axe.
    """
    if points is None:
        assert center is not None and radius is not None
    if center is None or radius is None:
        assert points is not None
    if center is None:
        center = points.mean(axis=0, keepdims=True)
    if radius is None:
        radius = points - center
        radius = np.max(np.abs(radius)) * ratio

    xroot, yroot, zroot = center[0, 0], center[0, 1], center[0, 2]
    ax.set_xlim3d([-radius + xroot, radius + xroot])
    ax.set_ylim3d([-radius + yroot, radius + yroot])
    ax.set_zlim3d([-radius + zroot, radius + zroot])

    return


def plot_3d_points(
        ax,
        points,
        indices=None,
        center=None,
        radius=None,
        add_labels=True,
        display_ticks=True,
        remove_planes=[],
        marker='o',
        color='k',
        size=50,
        alpha=1,
        set_limits=False):
    """
    Scatter plot of 3D points.

    points are of shape [3*N_points] or [N_points, 3]
    """
    points = points[indices, :] if indices is not None else points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=marker, c=color,
               s=size, alpha=alpha)
    if set_limits:
        set_3d_axe_limits(ax, points, center, radius)
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # remove tick labels or planes
    if not display_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

    white = (1.0, 1.0, 1.0, 1.0)
    if 'x' in remove_planes:
        ax.w_xaxis.set_pane_color(white)
    if 'y' in remove_planes:
        ax.w_xaxis.set_pane_color(white)
    if 'z' in remove_planes:
        ax.w_xaxis.set_pane_color(white)

    return


def plot_lines(
        ax,
        points,
        connections,
        dimension,
        lw=4,
        c='k',
        linestyle='-',
        alpha=0.8,
        add_index=False):
    """
    Plot 2D/3D lines given points and connection.

    connections are of shape [n_lines, 2]
    """
    if add_index:
        for idx in range(len(points)):
            if dimension == 2:
                x, y = points[idx][0], points[idx][1]
                ax.text(x, y, str(idx))
            elif dimension == 3:
                x, y, z = points[idx][0], points[idx][1], points[idx][2]
                ax.text(x, y, z, str(idx))

    connections = connections.reshape(-1, 2)
    for connection in connections:
        x = [points[connection[0]][0], points[connection[1]][0]]
        y = [points[connection[0]][1], points[connection[1]][1]]
        if dimension == 3:
            z = [points[connection[0]][2], points[connection[1]][2]]
            line, = ax.plot(x, y, z, lw=lw, c=c, linestyle=linestyle, alpha=alpha)
        else:
            line, = ax.plot(x, y, lw=lw, c=c, linestyle=linestyle, alpha=alpha)

    return line


def plot_3d_bbox(
        ax,
        bbox_3d_projected,
        color=None,
        linestyle='-',
        add_index=False):
    """
    Draw the projected edges of a 3D cuboid.
    """
    c = np.random.rand(3) if color is None else color
    plot_lines(
        ax,
        bbox_3d_projected,
        plot_3d_bbox.connections,
        dimension=2,
        c=c,
        linestyle=linestyle,
        add_index=add_index
    )
    return


def plot_2d_bbox(
        ax,
        bbox_2d,
        color=None,
        score=None,
        label=None,
        linestyle='-'):
    """
    Draw a 2D bounding box.
    
    bbox_2d in the format [x1, y1, x2, y2]
    """
    c = np.random.rand(3) if color is None else color
    x1, y1, x2, y2 = bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3],
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    plot_lines(ax, points, plot_2d_bbox.connections, dimension=2, c=c, linestyle=linestyle)
    if score is not None and label is not None:
        string = "({:.2f}, {:d})".format(score, label)
        ax.text((x1 + x2) * 0.5, (y1 + y2) * 0.5, string, bbox=dict(facecolor='red', alpha=0.2))

    return ax


def plot_scene_3dbox(
        ax, points_pred, points_gt=None, color='r'):
    """
    Plot the comparison of predicted 3d bounding boxes and ground truth ones.
    """
    preds = points_pred.copy()

    # add the root translation
    preds[:, 1:, ] = preds[:, 1:, ] + preds[:, [0], ]
    if points_gt is not None:
        gts = points_gt.copy()
        gts[:, 1:, ] = gts[:, 1:, ] + gts[:, [0], ]
        all_points = np.concatenate([preds, gts], axis=0).reshape(-1, 3)
    else:
        all_points = preds.reshape(-1, 3)

    for pred in preds:
        plot_3d_points(ax, pred, color=color, size=15)
        plot_lines(ax, pred[1:, ], plot_3d_bbox.connections, dimension=3, c=color)
    if points_gt is not None:
        for gt in gts:
            plot_3d_points(ax, gt, color='k', size=15)
            plot_lines(ax, gt[1:, ], plot_3d_bbox.connections, dimension=3, c='k')
    set_3d_axe_limits(ax, all_points)

    return ax


def draw_pose_vecs(ax, pose_vecs=None, color='black'):
    """
    Add pose vectors to a 3D matplotlib axe.
    """
    if pose_vecs is None:
        return
    for pose_vec in pose_vecs:
        x, y, z, pitch, yaw, roll = pose_vec
        string = "({:.2f}, {:.2f}, {:.2f})".format(pitch, yaw, roll)
        # add some random noise to the text location so that they do not overlap
        nl = 0.02  # noise level
        ax.text(
            x * (1 + np.random.randn() * nl),
            y * (1 + np.random.randn() * nl),
            z * (1 + np.random.randn() * nl),
            string,
            color=color
        )

    return


## static variables implemented as function attributes
plot_3d_bbox.connections = np.array(
    [[0, 1],
     [0, 2],
     [1, 3],
     [2, 3],
     [4, 5],
     [5, 7],
     [4, 6],
     [6, 7],
     [0, 4],
     [1, 5],
     [2, 6],
     [3, 7]])
plot_2d_bbox.connections = np.array(
    [[0, 1],
     [1, 2],
     [2, 3],
     [3, 0]])

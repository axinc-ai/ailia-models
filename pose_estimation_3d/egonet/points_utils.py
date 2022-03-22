import numpy as np


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

import numpy as np
import open3d as o3d


def draw_shifted_pts(mesh_name, pts, weights=None):
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    pred_joints = o3d.geometry.PointCloud()
    pred_joints.points = o3d.utility.Vector3dVector(pts)
    if weights is None:
        color_joints = [[1.0, 0.0, 0.0] for i in range(len(pts))]
    else:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('YlOrRd')
        # weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        # weights = 1 / (1 + np.exp(-weights))
        color_joints = cmap(weights.squeeze())
        color_joints = color_joints[:, :-1]
    pred_joints.colors = o3d.utility.Vector3dVector(color_joints)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    vis.add_geometry(pred_joints)

    param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    # vis.run()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    # param = ctr.convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image

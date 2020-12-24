import sys, os
import time

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

import open3d as o3d

# import binvox_rw
from utils import binvox_rw


def _normalize_obj(mesh_v):
    dims = [
        max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
        max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
        max(mesh_v[:, 2]) - min(mesh_v[:, 2])
    ]
    scale = 1.0 / max(dims)
    pivot = np.array([
        (min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, min(mesh_v[:, 1]),
        (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2
    ])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale


def _get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = set()
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.add(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(neighbor_ids)
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def _get_geo_edges(surface_geodesic, remesh_obj_v):
    edge_index = []
    surface_geodesic += 1.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= 0.06).squeeze(1)
        if len(geodesic_ball_samples) > 10:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, 10, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def _add_self_loops(
        edge_index, num_nodes=None):
    N = np.max(edge_index) + 1 if num_nodes is None else num_nodes
    loop_index = np.arange(N)
    loop_index = np.repeat(np.expand_dims(loop_index, 0), 2, axis=0)
    edge_index = np.concatenate([edge_index, loop_index], axis=1)
    return edge_index


def _calc_surface_geodesic(mesh):
    # We denselu sample 4000 points to be more accuracy.
    samples = mesh.sample_points_poisson_disk(number_of_points=4000)
    pts = np.asarray(samples.points)
    pts_normal = np.asarray(samples.normals)

    time1 = time.time()
    N = len(pts)
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    verts_nn = np.argsort(verts_dist, axis=1)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for p in range(N):
        nn_p = verts_nn[p, 1:6]
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
        norm_p = np.linalg.norm(pts_normal[p])
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
        nn_p = nn_p[cos_similar > -0.5]
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]
    [dist, _] = dijkstra(conn_matrix, directed=False, indices=range(N),
                         return_predecessors=True, unweighted=False)

    # replace inf distance with euclidean distance + 8
    # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
    inf_pos = np.argwhere(np.isinf(dist))
    if len(inf_pos) > 0:
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    verts = np.array(mesh.vertices)
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
    time2 = time.time()

    print('surface geodesic calculation: {} seconds'.format((time2 - time1)))
    return surface_geodesic


def create_single_data(mesh_filaname):
    """
    create input data for the network. The data is wrapped by Data structure in pytorch-geometric library
    :param mesh_filaname: name of the input mesh
    :return: wrapped data, voxelized mesh, and geodesic distance matrix of all vertices
    """
    mesh = o3d.io.read_triangle_mesh(mesh_filaname)
    mesh.compute_triangle_normals()
    mesh_v = np.asarray(mesh.vertices)
    mesh_vn = np.asarray(mesh.vertex_normals)
    mesh_f = np.asarray(mesh.triangles)

    mesh_v, translation_normalize, scale_normalize = _normalize_obj(mesh_v)

    # vertices
    v = np.concatenate((mesh_v, mesh_vn), axis=1)
    v = v.astype(np.float32)

    # topology edges
    print("     gathering topological edges.")
    tpl_e = _get_tpl_edges(mesh_v, mesh_f).T
    tpl_e = _add_self_loops(tpl_e, num_nodes=v.shape[0])
    tpl_e = tpl_e.astype(np.int64)

    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")
    surface_geodesic = _calc_surface_geodesic(mesh)

    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = _get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = _add_self_loops(geo_e, num_nodes=v.shape[0])
    geo_e = geo_e.astype(np.int64)

    # batch
    batch = np.zeros(len(v), dtype=np.int64)

    # voxel
    file_binvox = mesh_filaname.replace(".obj", "_normalized.binvox")
    file_normalized = mesh_filaname.replace(".obj", "_normalized.obj")
    if not os.path.exists(file_binvox):
        mesh_normalized = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh_v),
            triangles=o3d.utility.Vector3iVector(mesh_f))
        o3d.io.write_triangle_mesh(file_normalized, mesh_normalized)

        if sys.platform == "win32":
            os.system("binvox.exe -d 88 " + file_normalized)
        else:
            os.system("./binvox -d 88 -pb " + file_normalized)

    with open(file_binvox, 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)

    data = dict(
        batch=batch, pos=v[:, 0:3],
        tpl_edge_index=tpl_e, geo_edge_index=geo_e,
    )
    return data, vox, surface_geodesic, translation_normalize, scale_normalize


def inside_check(pts, vox):
    """
    Check where points are inside or outside the mesh based on its voxelization.
    :param pts: points to be checked
    :param vox: voxelized mesh
    :return: internal points, and index of them in the input array.
    """
    vc = (pts - vox.translate) / vox.scale * vox.dims[0]
    vc = np.round(vc).astype(int)
    ind1 = np.logical_and(np.all(vc >= 0, axis=1), np.all(vc < 88, axis=1))
    vc = np.clip(vc, 0, 87)
    ind2 = vox.data[vc[:, 0], vc[:, 1], vc[:, 2]]
    ind = np.logical_and(ind1, ind2)
    pts = pts[ind]
    return pts, np.argwhere(ind).squeeze()


def sample_on_bone(p_pos, ch_pos):
    """
    sample points on a bone
    :param p_pos: parent joint position
    :param ch_pos: child joint position
    :return: a array of samples on this bone.
    """
    ray = ch_pos - p_pos
    bone_length = np.sqrt(np.sum((p_pos - ch_pos) ** 2))
    num_step = np.round(bone_length / 0.01)
    i_step = np.arange(1, num_step + 1)
    unit_step = ray / (num_step + 1e-30)
    unit_step = np.repeat(unit_step[np.newaxis, :], num_step, axis=0)
    res = p_pos + unit_step * i_step[:, np.newaxis]
    return res


def meanshift_cluster(pts_in, bandwidth, weights=None, max_iter=20):
    """
    Meanshift clustering
    :param pts_in: input points
    :param bandwidth: bandwidth
    :param weights: weights per pts indicting its importance in the clustering
    :return: points after clustering
    """
    diff = 1e10
    num_iter = 1
    while diff > 1e-3 and num_iter < max_iter:
        Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
        K = np.maximum(bandwidth ** 2 - Y, np.zeros(Y.shape))
        if weights is not None:
            K = K * weights
        row_sums = K.sum(axis=0, keepdims=True)
        P = K / (row_sums + 1e-10)
        P = P.transpose()
        pts_in_prim = 0.3 * (np.matmul(P, pts_in) - pts_in) + pts_in
        diff = np.sqrt(np.sum((pts_in_prim - pts_in) ** 2))
        pts_in = pts_in_prim
        num_iter += 1
    return pts_in


def nms_meanshift(pts_in, density, bandwidth):
    """
    NMS to extract modes after meanshift. Code refers to sci-kit-learn.
    :param pts_in: input points
    :param density: density at each point
    :param bandwidth: bandwidth used in meanshift. Used here as neighbor region for NMS
    :return: extracted clusters.
    """
    Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
    sorted_ids = np.argsort(density)[::-1]
    unique = np.ones(len(sorted_ids), dtype=np.bool)
    dist = np.sqrt(Y)
    for i in sorted_ids:
        if unique[i]:
            neighbor_idxs = np.argwhere(dist[:, i] <= bandwidth)
            unique[neighbor_idxs.squeeze()] = 0
            unique[i] = 1  # leave the current point as unique
    pts_in = pts_in[unique]
    return pts_in


def flip(pred_joints):
    """
    symmetrize the predicted joints by reflecting joints on the left half space to the right
    :param pred_joints: raw predicted joints
    :return: symmetrized predicted joints
    """
    pred_joints_left = pred_joints[np.argwhere(pred_joints[:, 0] < -2e-2).squeeze(), :]
    pred_joints_middle = pred_joints[np.argwhere(np.abs(pred_joints[:, 0]) <= 2e-2).squeeze(), :]

    if pred_joints_left.ndim == 1:
        pred_joints_left = pred_joints_left[np.newaxis, :]
    if pred_joints_middle.ndim == 1:
        pred_joints_middle = pred_joints_middle[np.newaxis, :]

    pred_joints_middle[:, 0] = 0.0
    pred_joints_right = np.copy(pred_joints_left)
    pred_joints_right[:, 0] = -pred_joints_right[:, 0]
    pred_joints_res = np.concatenate((pred_joints_left, pred_joints_middle, pred_joints_right), axis=0)
    side_indicator = np.concatenate(
        (-np.ones(len(pred_joints_left)), np.zeros(len(pred_joints_middle)), np.ones(len(pred_joints_right))), axis=0)
    return pred_joints_res, side_indicator

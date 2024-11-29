import numpy as np


def umeyama(src, dst, estimate_scale=True):
    """
    Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    Reference
    Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return np.array(T[:2])


def invert(mat):
    """
    returns inverted Affine2DMat
    """
    ((a, b, c), (d, e, f)) = mat
    D = a * e - b * d
    D = 1.0 / D if D != 0.0 else 0.0
    a, b, c, d, e, f = (
        e * D,
        -b * D,
        (b * f - e * c) * D,
        -d * D,
        a * D,
        (d * c - a * f) * D,
    )

    return np.array(((a, b, c), (d, e, f)), dtype=np.float32)


def transform_points(mat, points):
    if not isinstance(points, np.ndarray):
        points = np.float32(points)

    dtype = points.dtype

    points = np.pad(points, ((0, 0), (0, 1)), constant_values=(1,), mode="constant")

    return (
        np.matmul(np.concatenate([mat, [[0, 0, 1]]], 0), points.T)
        .T[:, :2]
        .astype(dtype)
    )

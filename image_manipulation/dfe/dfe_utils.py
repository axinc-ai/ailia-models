import cv2
import numpy as np
import imageio


def estimate_for_model(pts1, pts2, weights):
    """Forward pass.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): points in second image
        weights (tensor): estimated weights

    Returns:
        tensor: estimated fundamental matrix
    """

    mask = np.ones(3)
    mask[-1] = 0

    def normalize(pts, weights):
        """Normalize points based on weights.

        Args:
            pts (tensor): points
            weights (tensor): estimated weights

        Returns:
            tensor: normalized points
        """

        denom = weights.sum(axis=1)

        center = np.sum(pts * weights, 1) / denom
        dist = pts - center[:, np.newaxis, :]
        dist = dist[:, :, :2]
        dist = np.power(dist, 2)
        dist = np.sum(dist, axis=2)
        dist = np.sqrt(dist)
        dist = dist[:, :, np.newaxis]
        meandist = (weights * dist).sum(axis=1) / denom
        meandist = np.squeeze(meandist, 1)

        scale = 1.4142 / meandist

        transform = np.zeros((pts.shape[0], 3, 3))

        transform[:, 0, 0] = scale
        transform[:, 1, 1] = scale
        transform[:, 2, 2] = 1
        transform[:, 0, 2] = -center[:, 0] * scale
        transform[:, 1, 2] = -center[:, 1] * scale

        pts_out = transform @ pts.transpose(0, 2, 1)

        return pts_out, transform

    def weighted_svd(pts1, pts2, weights):
        """Solve homogeneous least squares problem and extract model.

        Args:
            pts1 (tensor): points in first image
            pts2 (tensor): points in second image
            weights (tensor): estimated weights

        Returns:
            tensor: estimated fundamental matrix
        """

        weights = np.squeeze(weights, 1)[:, :, np.newaxis]

        pts1n, transform1 = normalize(pts1, weights)
        pts2n, transform2 = normalize(pts2, weights)

        t1 = (pts1n[:, 0])[:, np.newaxis, :] * pts2n
        t2 = (pts1n[:, 1])[:, np.newaxis, :] * pts2n
        p = np.concatenate([t1, t2, pts2n], axis=1).transpose(0, 2, 1)

        X = p * weights

        out_batch = []

        for batch in range(X.shape[0]):
            # solve homogeneous least squares problem
            _, _, V = np.linalg.svd(X[batch])
            V = V.T
            F = V[:, -1].reshape(3, 3)

            # model extractor
            U, S, V = np.linalg.svd(F)
            V = V.T

            F_projected = U @ (np.diag(S * mask)) @ V.T
            F_projected = F_projected[np.newaxis, :, :]

            out_batch.append(F_projected)

        out = np.concatenate(out_batch, axis=0)
        out = (transform1.transpose(0, 2, 1) @ out) @ transform2

        return out

    out = weighted_svd(pts1, pts2, weights)

    return out


def rescale_and_expand(pts):
    def normalize(pts):
        """Normalizes the input points to [-1, 1]^2 and transforms them to homogenous coordinates.
        Args:
            pts (tensor): input points

        Returns:
            tensor: transformed points
            tensor: transformation
        """
        ones = np.ones((pts.shape[0], pts.shape[1], 1))

        pts = np.concatenate([pts, ones], 2)

        center = np.mean(pts, axis=1)
        dist = pts - center[:, np.newaxis, :]

        dist = dist[:, :, :2]
        dist = np.power(dist, 2)
        dist = np.sum(dist, axis=2)
        dist = np.sqrt(dist)
        dist = np.mean(dist, axis=1)
        meandist = dist

        scale = 1.0 / meandist
        scale = scale[0]

        transform = np.zeros((pts.shape[0], 3, 3))

        transform[:, 0, 0] = scale
        transform[:, 1, 1] = scale
        transform[:, 2, 2] = 1
        transform[:, 0, 2] = -center[:, 0] * scale
        transform[:, 1, 2] = -center[:, 1] * scale

        pts_out = transform @ pts.transpose((0, 2, 1))

        return pts_out, transform

    """Forward pass.
    Args:
        pts (tensor): point correspondences

    Returns:
        tensor: transformed points in first image
        tensor: transformed points in second image
        tensor: transformtion (first image)
        tensor: transformtion (second image)
    """
    pts1, transform1 = normalize(pts[:, :, :2])
    pts2, transform2 = normalize(pts[:, :, 2:])

    return pts1, pts2, transform1, transform2


def robust_symmetric_epipolar_distance(pts1, pts2, fundamental_mat, gamma=0.5):
    """Robust symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix
        gamma (float, optional): Defaults to 0.5. robust parameter

    Returns:
        tensor: robust symmetric epipolar distance
    """

    sed = symmetric_epipolar_distance(pts1, pts2, fundamental_mat)
    ret = np.clip(sed, None, gamma)
    ret = ret[:, np.newaxis, :]

    return ret


def symmetric_epipolar_distance(pts1, pts2, fundamental_mat):
    """Symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix

    Returns:
        tensor: symmetric epipolar distance
    """

    line_1 = pts1 @ fundamental_mat
    line_2 = pts2 @ fundamental_mat.transpose(0, 2, 1)
    scalar_product = np.sum(pts2 * line_1, axis=2)
    ret = np.absolute(scalar_product) * (
        1 / np.linalg.norm(line_1[:, :, :2], axis=2) + 1 / np.linalg.norm(line_2[:, :, :2], axis=2)
    )

    return ret


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!'%out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice

    
def load_image(img_file, show_zoom_info=True):
    img_ori = cv2.imread(img_file)
    return img_ori, (1.0, 1.0), img_ori


def get_sift_features(img_ori, zoom_xy):
    sift_num = 2000
    sift = cv2.SIFT_create(nfeatures=sift_num, contrastThreshold=1e-5)
    kp, des = sift.detectAndCompute(img_ori, None)
    x_all = np.array([p.pt for p in kp])
    x_all = (x_all * np.array([[zoom_xy[0], zoom_xy[1]]])).astype(np.float32)
    if x_all.shape[0] != sift_num:
        choice = crop_or_pad_choice(x_all.shape[0], sift_num, shuffle=True)
        x_all = x_all[choice]
        des = des[choice]
    return x_all, des


def do_matching(sift_matcher, des1, des2, sift_kps_ii, sift_kps_jj):
    def normalize(pts, W, H):
        ones = np.ones((pts.shape[0], pts.shape[1], 1))
        pts = np.concatenate([pts, ones], 2)
        T = np.array([[2./W, 0., -1.], [0., 2./H, -1.], [0., 0., 1.]], dtype=pts.dtype)
        T = T[np.newaxis, :, :]
        pts_out = T @ pts.transpose(0, 2, 1)
        return pts_out, T

    def normalize_and_expand_hw(pts):
        image_size = [2710, 3384, 3] # omitted
        H, W = image_size[0], image_size[1]
        pts1, T1 = normalize(pts[:,:,:2], H, W)
        pts2, T2 = normalize(pts[:,:,2:], H, W)

        return pts1, pts2, T1, T2

    def get_input(matches_xy_ori):
        """ get model input
        return: 
            weight_in: [B, N, 4] matching
            pts1, pts2: matching points
            T1, T2: camera intrinsic matrix
        """
        pts = matches_xy_ori
        pts1, pts2, T1, T2 = normalize_and_expand_hw(pts)
        pts1 = pts1.transpose(0, 2, 1)
        pts2 = pts2.transpose(0, 2, 1)
        weight_in = np.concatenate([(pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2], 2).transpose(0, 2, 1) # [0, 1]

        return weight_in, pts1, pts2, T1, T2

    try:
        # another option is https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py#L309
        matches = sift_matcher.knnMatch(des1, des2, k=2)
    except Exception as e:
        print('error')
        exit()

    # store all the good matches as per Lowe's ratio test.
    good = []
    all_m = []
    quality_good = []
    quality_all = []
    for m, n in matches:
        all_m.append(m)
        if m.distance < 0.8 * n.distance:
            good.append(m)
            quality_good.append([m.distance, m.distance / n.distance])
        quality_all.append([m.distance, m.distance / n.distance])

    good_ij = [[mat.queryIdx for mat in good], [mat.trainIdx for mat in good]]
    all_ij = [[mat.queryIdx for mat in all_m], [mat.trainIdx for mat in all_m]]

    good_ij = np.asarray(good_ij, dtype=np.int32).T.copy()
    all_ij = np.asarray(all_ij, dtype=np.int32).T.copy()
    quality_good = np.asarray(quality_good, dtype=np.float32).copy()
    quality_all = np.asarray(quality_all, dtype=np.float32).copy()

    match_quality_good = np.hstack(
        (sift_kps_ii[good_ij[:, 0]], sift_kps_jj[good_ij[:, 1]], quality_good)
    )  # [[x1, y1, x2, y2, dist_good, ratio_good]]
    match_quality_all = np.hstack(
        (sift_kps_ii[all_ij[:, 0]], sift_kps_jj[all_ij[:, 1]], quality_all)
    )  # [[x1, y1, x2, y2, dist_good, ratio_good]]
    
    ### format for dataset ###
    match_qualitys = [match_quality_all, match_quality_good]

    matches_all = match_qualitys[0][:, :4]
    #matches_all = scale_points(matches_all, zoom_xy, loop_length=4)
    choice_all = crop_or_pad_choice(matches_all.shape[0], 2000, shuffle=True)
    matches_all_padded = matches_all[choice_all]

    matches_good = match_qualitys[1][:, :4]
    #matches_good = scale_points(matches_good, zoom_xy, loop_length=4)
    choice_good = crop_or_pad_choice(matches_good.shape[0], 1000, shuffle=True,)
    matches_good_padded = matches_good[choice_good]

    matches_all, matches_good = matches_all_padded, matches_good_padded

    #K, K_inv = add_scaled_K(K_ori, zoom_xy=zoom_xy)

    ### format for model input ###
    if_SIFT = True
    if if_SIFT:
        matches_use = matches_good
    
    N_corres = matches_use.shape[0]  # 1311 for matches_good, 2000 for matches_all
    x1, x2 = (matches_use[:, :2], matches_use[:, 2:])
    
    x1 = x1[np.newaxis, :, :]
    x2 = x2[np.newaxis, :, :]
    matches_use_ori = np.concatenate([x1, x2], 2)

    weight_in, pts1, pts2, T1, T2 = get_input(matches_use_ori)

    matches_use_ori = matches_use_ori.transpose(0, 2, 1) 

    return matches_use_ori, weight_in, pts1, pts2, T1, T2, x1, x2
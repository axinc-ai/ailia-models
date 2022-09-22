import numpy as np
import cv2
import matplotlib.pyplot as plt

# for pytorch-superpoint
from functional import grid_sample  # noqa
from scipy.special import softmax


CONFIG = {
    'detection_threshold': 0.015,
    'nms': 4,
    'subpixel': {
        'patch_size': 5
    }
}


class DepthToSpaceNumpy():
    def __init__(self, block_size):
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.transpose((0, 2, 3, 1))
        (batch_size, d_height, d_width, d_depth) = output.shape
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape([batch_size, d_height, d_width, self.block_size_sq, s_depth])
        spl = np.split(t_1, self.block_size, 3)
        stack = [t_t.reshape([batch_size, d_height, s_width, s_depth]) for t_t in spl]
        output = np.stack(stack, axis=0)
        output = np.swapaxes(output, 0, 1)
        output = output.transpose((0,2,1,3,4))
        output = output.reshape([batch_size, s_height, s_width, s_depth])
        output = output.transpose((0, 3, 1, 2))
        return output

def flattenDetection(semi, tensor=False):
    '''
    Flatten detection output

    :param semi:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        np (1, H, C)
        :or
        tensor (batch_size, 65, Hc, Wc)

    '''
    batch = False
    if len(semi.shape) == 4:
        batch = True
        batch_size = semi.shape[0]
    if batch:
        #dense = nn.functional.softmax(semi, dim=1) # [batch, 65, Hc, Wc]
        dense = softmax(semi, axis=1)
        # Remove dustbin.
        nodust = dense[:, :-1, :, :]
    else:
        #dense = nn.functional.softmax(semi, dim=0) # [65, Hc, Wc]
        dense = softmax(semi, axis=0)
        nodust = dense[:-1, :, :].unsqueeze(0)
    # Reshape to get full resolution heatmap.
    # heatmap = flatten64to1(nodust, tensor=True) # [1, H, W]
    depth2space = DepthToSpaceNumpy(8)
    heatmap = depth2space.forward(nodust)
    heatmap = heatmap.squeeze(0) if not batch else heatmap
    return heatmap

def heatmap_to_pts(heatmap):
    heatmap_np = heatmap[0][0]
    pts_nms_batch = [getPtsFromHeatmap(heatmap_np)]
    return pts_nms_batch

def getPtsFromHeatmap(heatmap):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''
    conf_thresh = CONFIG['detection_threshold']
    nms_dist = CONFIG['nms']
    border_remove = 4  # Remove points this close to the border.

    heatmap = heatmap.squeeze()
    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys # abuse of ys, xs
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts

def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
        3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
        0 : Empty or suppressed.
        1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
        in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
        H - Image height.
        W - Image width.
        dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
        nmsed_corners - 3xN numpy matrix with surviving corners.
        nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def soft_argmax_points(heatmap, pts, patch_size=5):
    """
    input:
        pts: tensor [N x 2]
    """
    ##### check not take care of batch #####
    # print("not take care of batch! only take first element!")
    pts = pts[0].transpose().copy()
    patches = extract_patch_from_points(heatmap, pts, patch_size=patch_size)
    patches = np.stack(patches)
    patches = patches[np.newaxis, :, :, :]

    # norm patches
    patches = norm_patches(patches)
    patches = do_log(patches)
    dxdy = soft_argmax_2d(patches, normalized_coordinates=False)

    points = pts
    points[:,:2] = points[:,:2] + dxdy.squeeze() - patch_size//2
    pts_subpixel = [points.transpose().copy()]
    return pts_subpixel.copy()

def desc_to_sparseDesc(pts_nms_batch, out_desc):
    desc_sparse_batch = [sample_desc_from_points(out_desc, pts) for pts in pts_nms_batch]
    return desc_sparse_batch

def sample_desc_from_points(coarse_desc, pts):
    cell = 8

    H, W = coarse_desc.shape[2]*cell, coarse_desc.shape[3]*cell
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        samp_pts = pts[:2, :].copy()
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
        samp_pts = np.swapaxes(samp_pts, 0, 1)
        samp_pts = np.ascontiguousarray(samp_pts)
        samp_pts = samp_pts.reshape([1, 1, -1, 2])
        samp_pts = samp_pts.astype(np.float32)

        #desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        desc = grid_sample(coarse_desc, samp_pts, align_corners=True)

        desc = desc.reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc


"""
From: pytorch-superpoint/evaluation.py
"""

def getInliers(matches, H, epi=3, verbose=False):
    """
    input:
        matches: numpy (n, 4(x1, y1, x2, y2))
        H (ground truth homography): numpy (3, 3)
    """
    from evaluations.detector_evaluation import warp_keypoints
    # warp points 
    warped_points = warp_keypoints(matches[:, :2], H) # make sure the input fits the (x,y)

    # compute point distance
    norm = np.linalg.norm(warped_points - matches[:, 2:4],
                            ord=None, axis=1)
    inliers = norm < epi
    if verbose:
        print("Total matches: ", inliers.shape[0], ", inliers: ", inliers.sum(),
                            ", percentage: ", inliers.sum() / inliers.shape[0])

    return inliers

def flipArr(arr):
    return arr.max() - arr

def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img

def draw_matches_cv(data, matches, plot_points=True):
    if plot_points:
        keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
        keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    else:
        matches_pts = data['matches']
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in matches_pts]
        keypoints2 = [cv2.KeyPoint(p[2], p[3], 1) for p in matches_pts]

    inliers = data['inliers'].astype(bool)
    img1 = to3dim(data['image1'])
    img2 = to3dim(data['image2'])
    img1 = np.concatenate([img1, img1, img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    return cv2.drawMatches(np.uint8(img1), keypoints1, np.uint8(img2), keypoints2, matches,
                            None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))

def get_random_m(matches, ratio):
    ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))               
    return matches[ran_idx], ran_idx


"""
From: pytorch-superpoint/evaluations/descriptor_evaluation.py
"""
def compute_homography(data, keep_k_points=1000, correctness_thresh=3, orb=False, shape=(240,320)):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    keypoints = data['prob'][:,[1, 0]]
    warped_keypoints = data['warped_prob'][:,[1, 0]]

    desc = data['desc']
    warped_desc = data['warped_desc']

    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if orb:
        desc = desc.astype(np.uint8)
        warped_desc = warped_desc.astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    cv2_matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in cv2_matches])
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in cv2_matches])
    m_dist = np.array([m.distance for m in cv2_matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]
    matches = np.hstack((m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]]))

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(m_keypoints[:, [1, 0]],
                                    m_warped_keypoints[:, [1, 0]],
                                    cv2.RANSAC)                  
    inliers = inliers.flatten()

    # Compute correctness
    if H is None:
        correctness = 0
        H = np.identity(3)
        print("no valid estimation")
    else:
        corners = np.array([[0, 0, 1],
                            [0, shape[0] - 1, 1],
                            [shape[1] - 1, 0, 1],
                            [shape[1] - 1, shape[0] - 1, 1]])      
        warped_corners = np.dot(corners, np.transpose(H))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]

    return {'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,  # cv2.match
            'cv2_matches': cv2_matches,
            'mscores': m_dist/(m_dist.max()), # normalized distance
            'inliers': inliers,
            'homography': H}


"""
From: pytorch-superpoint/utils/draw.py
"""

def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
    '''

    :param img:
        image:
        numpy [H, W]
    :param corners:
        Points
        numpy [N, 2]
    :param color:
    :param radius:
    :param s:
    :return:
        overlaying image
        numpy [H, W]
    '''
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
    return img

def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        fig, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()

def draw_matches(rgb1, rgb2, match_pairs, lw = 0.5, color='g', if_fig=True,
                filename='matches.png', show=False):
    '''

    :param rgb1:
        image1
        numpy (H, W)
    :param rgb2:
        image2
        numpy (H, W)
    :param match_pairs:
        numpy (keypoiny1 x, keypoint1 y, keypoint2 x, keypoint 2 y)
    :return:
        None
    '''
    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=rgb1.dtype)
    canvas[:h1, :w1] = rgb1[:,:,np.newaxis]
    canvas[:h2, w1:] = rgb2[:,:,np.newaxis]
    # fig = plt.figure(frameon=False)
    if if_fig:
        fig = plt.figure(figsize=(15,5))
    plt.axis("off")
    plt.imshow(canvas, zorder=1)

    xs = match_pairs[:, [0, 2]]
    xs[:, 1] += w1
    ys = match_pairs[:, [1, 3]]

    alpha = 1
    sf = 5
    # lw = 0.5
    # markersize = 1
    markersize = 2

    plt.plot(
        xs.T, ys.T,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        marker='o',
        markersize=markersize,
        fillstyle='none',
        color=color,
        zorder=2,
        # color=[0.0, 0.8, 0.0],
    );
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    plt.clf()


"""
From: pytorch-superpoint/utils/losses.py
"""

def extract_patch_from_points(heatmap, points, patch_size=5):
    """
    this function works in numpy
    """
    # numpy
    heatmap = heatmap.squeeze()  # [H, W]
    # padding
    pad_size = int(patch_size/2)
    heatmap = np.pad(heatmap, pad_size, 'constant')
    # crop it
    patches = []
    ext = lambda img, pnt, wid: img[pnt[1]:pnt[1]+wid, pnt[0]:pnt[0]+wid]
    for i in range(points.shape[0]):
        patch = ext(heatmap, points[i,:].astype(int), patch_size)
        patches.append(patch)
        
        # if i > 10: break
    # extract points
    return patches

def soft_argmax_2d(patches, normalized_coordinates=True):
    """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """
    m = SpatialSoftArgmax2dNumpy(normalized_coordinates=normalized_coordinates)
    coords = m.forward(patches)  # 1x4x2

    return coords

def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = patches.reshape([-1, 1, patch_size*patch_size])
    d = np.sum(patches, axis=-1)[:, :, np.newaxis] + 1e-6
    patches = patches/d
    patches = patches.reshape([-1, 1, patch_size, patch_size])
    return patches

def do_log(patches):
    patches[patches<0] = 1e-6
    patches_log = np.log(patches)
    return patches_log


"""
From: 
    torchgeometry.contrib.spatial_soft_argmax2d
    https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/contrib/spatial_soft_argmax2d.html
"""

from typing import Optional

class SpatialSoftArgmax2dNumpy():
    def __init__(self, normalized_coordinates: Optional[bool] = True) -> None:
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.eps: float = 1e-6

    def create_meshgrid(self, x, normalized_coordinates: Optional[bool]):
        _, _, height, width = x.shape
        if normalized_coordinates:
            xs = np.linspace(-1.0, 1.0, width)
            ys = np.linspace(-1.0, 1.0, height)
        else:
            xs = np.linspace(0, width - 1, width)
            ys = np.linspace(0, height - 1, height)
        output = np.meshgrid(xs, ys)
        output = (output[1], output[0])
        return output  # pos_y, pos_x

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        x = input.reshape([batch_size, channels, -1])

        # compute softmax with max substraction trick
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        exp_x_sum = 1.0 / (exp_x.sum(axis=-1, keepdims=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = self.create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y = np.sum((pos_y * exp_x) * exp_x_sum, axis=-1, keepdims=True)
        expected_x = np.sum((pos_x * exp_x) * exp_x_sum, axis=-1, keepdims=True)
        output = np.concatenate([expected_x, expected_y], axis=-1)
        output = output.reshape([batch_size, channels, 2])
        return output
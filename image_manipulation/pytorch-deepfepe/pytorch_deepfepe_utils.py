import argparse
import json
import os
import sys
import time

import ailia
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from detector_utils import load_image  # noqa: E402C
from image_utils import imread  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import get_base_parser, get_savepath, update_parser  # noqa: E402

logger = getLogger(__name__)


### Setting of parameters ###
config = {
    "data": {
        "sequence_length": 2,
        "delta_ij": 1,
        "good_num": 1000,
        "image": {
            "size": [2710, 3384, 3]
        }
    },
    "model": {
        "name": "GoodCorresNet_layers_deepF",
        "clamp_at": 0.02,
        "depth": 5
    }
}


# TODO: Eliminate pytorch dependency
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F


### Dataset ###
def R_to_q_np(matrix):
    """ https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L204
    This code uses a modification of the algorithm described in:
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    which is itself based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    Altered to work with the column vector convention instead of row vectors
    
    """
    m = matrix.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

    q = np.array(q, dtype=np.float32)
    q *= 0.5 / np.sqrt(t)
    if q[0] < 0.:
        q = -q
    return q.reshape(-1, 1)
    
class Dataset():
    def _load_and_resize_img(self, img_file, sizerHW):
        print('Load img: {}'.format(img_file))
        img_ori = cv2.imread(img_file)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        if [sizerHW[0], sizerHW[1]] == [img_ori.shape[0], img_ori.shape[1]]: # H, W
            return img_ori, (1., 1.)
        else:
            zoom_y = sizerHW[0] / img_ori.shape[0]
            zoom_x = sizerHW[1] / img_ori.shape[1]
            img = cv2.resize(img_ori, (sizerHW[1], sizerHW[0]))
            return img, (zoom_x, zoom_y)

    def _load_as_array(self, path, dtype=None):
        print('Load as array: {}'.format(path))
        array = np.load(path)
        if dtype is not None:
            return array.astype(dtype)
        else:
            return array

    def _crop_or_pad_choice(self, in_num_points, out_num_points, shuffle=False):
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

    def _scale_points(self, points, zoom_xy, loop_length=2):
        """
        # iteratively scale x, y, x, y, ...
        """
        for i in range(loop_length):
            points[:, i] = points[:, i]*zoom_xy[i%2]
        return points
        pass

    def _identity_Rt(self, dtype=np.float32):
        return np.hstack((np.eye(3, dtype=dtype), np.zeros((3, 1), dtype=dtype)))

    def _Rt_pad(self, Rt):
        # Padding 3*4 [R|t] to 4*4 [[R|t], [0, 1]]
        assert Rt.shape==(3, 4)
        return np.vstack((Rt, np.array([[0., 0., 0., 1.]], dtype=Rt.dtype)))

    def scale_P(self, P, sx, sy, if_print=False):
        if if_print:
            print(f'---scale_P - sx={sx}, sy={sy}')
        assert P.shape==(3, 4)
        out = np.copy(P)
        out[0] *= sx
        out[1] *= sy
        return out

    def add_scaled_K(self, sample, zoom_xy=[1,1]):
        """
        # scale calibration matrix based on img_zoom ratio. Add to the dict
        """
        if zoom_xy[0] != 1 or zoom_xy[1] != 1:
            print("note: scaled_K with zoom_xy = {}".format(zoom_xy))
        P_rect_ori = np.concatenate((sample['K_ori'], [[0], [0], [0]]), axis=1).astype(np.float32)
        P_rect_scale = self.scale_P(
            P_rect_ori, zoom_xy[0], zoom_xy[1]
        )
        K = P_rect_scale[:, :3]
        sample.update({
            'K': K,
            "K_inv": np.linalg.inv(K),
        })
        #logging.debug(f"K_ori: {sample['K_ori']}, type: {sample['K_ori'].dtype}, K: {sample['K']}, type: {K.dtype}")
        return sample

    def get_E_F(self, sample):
        """
        # add essential and fundamental matrix based on K.
        # *** must use the updated K!!!
        """
        relative_scene_pose = sample['relative_scene_poses'][1]
        sample["E"], sample["F"] = self.E_F_from_Rt_np(
            relative_scene_pose[:3, :3],
            relative_scene_pose[:3, 3:4],
            sample["K"],
        )
        return sample

    def E_F_from_Rt_np(self, R, t, K):
        """ Better use F instead of E """
        t_gt_x = self.skew_symmetric_np(t)
    #     print(t_gt_x, R_th)
        E_gt = t_gt_x@R
        if len(R.shape)==2:
            F_gt = np.linalg.inv(K).T @ E_gt @ np.linalg.inv(K)
        else:
            F_gt = np.linalg.inv(K).transpose(1, 2) @ E_gt @ np.linalg.inv(K)
        return E_gt, F_gt

    def skew_symmetric_np(self, v): # v: [3, 1] or [batch_size, 3, 1]
        if len(v.shape)==2:
            zero = np.zeros_like(v[0, 0])
            M = np.stack([
                zero, -v[2, 0], v[1, 0],
                v[2, 0], zero, -v[0, 0],
                -v[1, 0], v[0, 0], zero,
            ], axis=0)
            return M.reshape(3, 3)
        else:
            zero = np.zeros_like(v[:, 0, 0])
            M = np.stack([
                zero, -v[:, 2, 0], v[:, 1, 0],
                v[:, 2, 0], zero, -v[:, 0, 0],
                -v[:, 1, 0], v[:, 0, 0], zero,
            ], axis=1)
            return M.reshape(-1, 3, 3)

    def homo_np(self, x):
        # input: x [N, D]
        # output: x_homo [N, D+1]
        N = x.shape[0]
        x_homo = np.hstack((x, np.ones((N, 1), dtype=x.dtype)))
        return x_homo

    def get_virt_x1x2_np(self, im_shape, F_gt, K, pts1_virt_b, pts2_virt_b): ##  [RUI] TODO!!!!! Convert into seq loader!
        ## s.t. SHOULD BE ALL ZEROS: losses = utils_F.compute_epi_residual(pts1_virt_ori, pts2_virt_ori, F_gts, loss_params['clamp_at'])
        ## Reproject by minimizing distance to groundtruth epipolar lines
        pts1_virt, pts2_virt = cv2.correctMatches(F_gt, np.expand_dims(pts2_virt_b, 0), np.expand_dims(pts1_virt_b, 0))
        pts1_virt[np.isnan(pts1_virt)] = 0.
        pts2_virt[np.isnan(pts2_virt)] = 0.

        # nan1 = np.logical_and(
        #         np.logical_not(np.isnan(pts1_virt[:,:,0])),
        #         np.logical_not(np.isnan(pts1_virt[:,:,1])))
        # nan2 = np.logical_and(
        #         np.logical_not(np.isnan(pts2_virt[:,:,0])),
        #         np.logical_not(np.isnan(pts2_virt[:,:,1])))
        # _, midx = np.where(np.logical_and(nan1, nan2))
        # good_pts = len(midx)
        # while good_pts < num_pts_full:
        #     midx = np.hstack((midx, midx[:(num_pts_full-good_pts)]))
        #     good_pts = len(midx)
        # midx = midx[:num_pts_full]
        # pts1_virt = pts1_virt[:,midx]
        # pts2_virt = pts2_virt[:,midx]

        pts1_virt = self.homo_np(pts1_virt[0])
        pts2_virt = self.homo_np(pts2_virt[0])
        pts1_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
        pts2_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
        return pts1_virt_normalized, pts2_virt_normalized, pts1_virt, pts2_virt

    def get_virt_x1x2_grid(self, im_shape):
        step = 0.1
        sz1 = im_shape
        sz2 = im_shape
        xx, yy = np.meshgrid(np.arange(0, 1 , step), np.arange(0, 1, step))
        num_pts_full = len(xx.flatten())
        pts1_virt_b = np.float32(np.vstack((sz1[1]*xx.flatten(),sz1[0]*yy.flatten())).T)
        pts2_virt_b = np.float32(np.vstack((sz2[1]*xx.flatten(),sz2[0]*yy.flatten())).T)
        return pts1_virt_b, pts2_virt_b

    def prepare(self, frame_id = '000000'):
        sequence_length = config["data"]["sequence_length"]
        delta_ij = config["data"]["delta_ij"]
        sizerHW = [600, 800] # omitted

        frame = ['', '000000']

        scene = frame[0]
        frame_id = frame[1]
        frame_num = int(frame_id)

        K = self._load_as_array(os.path.join(scene, "cam.npy")).astype(np.float32).reshape((3, 3))
        poses = self._load_as_array(os.path.join(scene, "poses.npy"))
        poses = poses.astype(np.float32).reshape(-1, 3, 4)

        sample = {
            "K_ori": K,
            "scene": scene,
            "imgs": [],
            "ids": [],
            "frame_ids": [],
            "relative_scene_poses": []
        }

        for k in range(0, sequence_length):
            j = k * delta_ij + frame_num
            img_file_j = os.path.join(scene, ("%s.jpg" % ("%06d" % j)))
            sample["imgs"].append(img_file_j)
            if k == 0:
                sample["relative_scene_poses"].append(
                    self._identity_Rt(dtype=np.float32)
                )
            else:
                relative_scene_pose = np.linalg.inv(self._Rt_pad(poses[j])) @ self._Rt_pad(poses[frame_num])
                #if self.cam_id == "02":
                #    relative_scene_pose = (Rt_cam2_gt @ relative_scene_pose @ np.linalg.inv(Rt_cam2_gt))
                sample["relative_scene_poses"].append(relative_scene_pose)  # [4, 4]
            sample["frame_ids"].append(os.path.basename(img_file_j).replace(".jpg", ""))
            sample["ids"].append(j)

        imgs = []
        zoom_xys = []
        for img_file in sample['imgs']:
            image_rgb, zoom_xy = self._load_and_resize_img(img_file, sizerHW)
            img_totype = lambda x: x.astype('float32')
            imgs.append(img_totype(image_rgb))
            zoom_xys.append(zoom_xy)
        zoom_xy = zoom_xys[-1] #### assume images are in the same size
        sample.update({'imgs': imgs})

        sample = self.add_scaled_K(sample, zoom_xy=zoom_xy)
        sample = self.get_E_F(sample)

        pts1_virt_b, pts2_virt_b = self.get_virt_x1x2_grid(sizerHW)
        sample["pts1_virt_normalized"], sample["pts2_virt_normalized"], sample["pts1_virt"], sample["pts2_virt"] = \
        self.get_virt_x1x2_np(
            config["data"]["image"]["size"],
            sample["F"],
            sample["K"],
            pts1_virt_b,
            pts2_virt_b,
        )

        ids = sample["ids"]

        dump_ij_match_quality_file_name = os.path.join(sample["scene"], "ij_match_quality_{}-{}".format(ids[0], ids[1]))
        dump_ij_match_quality_files = [
            dump_ij_match_quality_file_name + "_all.npy",
            dump_ij_match_quality_file_name + "_good.npy",
        ]

        match_qualitys = []
        for dump_ij_match_quality_file in dump_ij_match_quality_files:
            if os.path.isfile(dump_ij_match_quality_file):
                match_qualitys.append(
                    self._load_as_array(dump_ij_match_quality_file).astype(np.float32)
                )
            else:
                print("NOT Find {}".format(dump_ij_match_quality_file))
                exit()

        matches_all = match_qualitys[0][:, :4]
        matches_all = self._scale_points(matches_all, zoom_xy, loop_length=4)
        choice_all = self._crop_or_pad_choice(
            matches_all.shape[0], 
            2000, 
            shuffle=True
        )
        matches_all = matches_all[choice_all]

        matches_good = match_qualitys[1][:, :4]
        matches_good = self._scale_points(matches_good, zoom_xy, loop_length=4)
        choice_good = self._crop_or_pad_choice(
            matches_good.shape[0],
            config["data"]["good_num"],
            shuffle=True
        )
        matches_good = matches_good[choice_good]

        matches_good_unique_nums = min(
            matches_good.shape[0],
            config["data"]["good_num"],
        )

        sample.update({
            "matches_all": matches_all,
            "matches_good": matches_good,
            "matches_good_unique_nums": matches_good_unique_nums
        })

        Rt_cam = np.linalg.inv(sample["relative_scene_poses"][1])
        R_cam = Rt_cam[:3, :3]
        t_cam = Rt_cam[:3, 3:4]
        q_cam = R_to_q_np(R_cam)
        Rt_scene = sample["relative_scene_poses"][1]
        R_scene = Rt_scene[:3, :3]
        t_scene = Rt_scene[:3, 3:4]
        q_scene = R_to_q_np(R_scene)
        sample.update(
            {
                "q_cam": q_cam,
                "t_cam": t_cam,
                "q_scene": q_scene,
                "t_scene": t_scene,
            }
        )

        return sample


### Extract features ###
class FeaturesExtractor():
    def __init__(self, if_SP=False):
        self.if_SP = if_SP

    def _normalize(self, pts, ones_b, W, H):
        ones = ones_b.expand(pts.size(0), pts.size(1), 1)
        T = torch.tensor([[2./W, 0., -1.], [0., 2./H, -1.], [0., 0., 1.]], device=pts.device, dtype=pts.dtype).unsqueeze(0).expand(pts.size(0), -1, -1)
        pts = torch.cat((pts, ones), 2)
        pts_out = T @ pts.permute(0,2,1)
        return pts_out, T

    def _normaliza_and_expand_hw(self, pts):
        image_size = [2710, 3384, 3] # omitted
        ones_b = Variable(torch.ones((1, 1, 1)), volatile=False)
        H, W = image_size[0], image_size[1]

        pts1, T1 = self._normalize(pts[:,:,:2], ones_b, H, W)
        pts2, T2 = self._normalize(pts[:,:,2:], ones_b, H, W)

        return pts1, pts2, T1, T2
    
    def extract(self, matches_good):
        if self.if_SP:
            # TODO:
            print('Not implemented.')
        else:
            matches_use = matches_good
            matches_use = torch.from_numpy(matches_use.astype(np.float32)).clone()
            matches_use = matches_use.unsqueeze(0)
        x1, x2 = (matches_use[:, :, :2], matches_use[:, :, 2:])

        matches_use_ori = torch.cat((x1, x2), 2)
        quality_use = None

        return matches_use_ori, quality_use

    def get_input(self, matches_xy_ori, quality):
        """ get model input
        return: 
            weight_in: [B, N, 4] matching
            pts1, pts2: matching points
            T1, T2: camera intrinsic matrix
        """
        pts = matches_xy_ori
        pts1, pts2, T1, T2 = self._normaliza_and_expand_hw(pts)
        pts1 = pts1.permute(0,2,1)
        pts2 = pts2.permute(0,2,1)
        weight_in = torch.cat(((pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2), 2).permute(0,2,1) # [0, 1]

        return weight_in, pts1, pts2, T1, T2


### Model ###
def set_nan2zero(tens, name = 'network'):
    mat_nans = (tens != tens)
    n_nans = mat_nans.sum()
    tens[mat_nans] = 0 
    return tens

  
def compute_epi_residual(pts1, pts2, F, clamp_at=0.5):
    l1 = pts2 @ F
    l2 = pts1 @ F.permute(0,2,1)
    dd = ((pts1*l1).sum(2))
    epi = 1e-6
    d = dd.abs()*(1/(l1[:,:,:2].norm(2,2) + epi) + 1/(l2[:,:,:2].norm(2,2) + epi)) # d1+d2, no squared
    out = torch.clamp(d, max=clamp_at)
    return out


class Fit(nn.Module):
    def __init__(self, is_cuda=True, is_test=False, if_cpu_svd=False, normalize_SVD=True):
        print('Fit params: ', is_cuda, is_test, if_cpu_svd, normalize_SVD)
        super(Fit, self).__init__()

        self.ones_b = Variable(torch.ones((1, 1, 1)).float())
        self.zero_b = Variable(torch.zeros((1, 1, 1)).float())
        self.T_b = torch.zeros(1, 3, 3).float()

        self.mask = Variable(torch.ones(3))
        self.mask[-1] = 0

        self.normalize_SVD = normalize_SVD
        self.if_cpu_svd = if_cpu_svd
        if self.if_cpu_svd:
            self.mask_cpu = self.mask.clone()

        if is_cuda:
            self.ones_b = self.ones_b.cuda()
            self.zero_b = self.zero_b.cuda()
            self.T_b = self.T_b.cuda()
            self.mask = self.mask.cuda()
        self.is_cuda = is_cuda

    def normalize(self, pts, weights):
        """ normalize the points to the weighted center """
        device = pts.device
        T = Variable(self.T_b.to(device).expand(pts.size(0), 3, 3)).clone()
        ones = self.ones_b.to(device).expand(pts.size(0), pts.size(1), 1)

        denom = weights.sum(1)

        ## get the center
        c = torch.sum(pts*weights,1)/denom
        newpts_ = (pts - c.unsqueeze(1))
        meandist = ((weights*(newpts_[:,:,:2].pow(2).sum(2).sqrt().unsqueeze(2))).sum(1)/denom).squeeze(1)

        scale = 1.4142/meandist

        T[:,0,0] = scale
        T[:,1,1] = scale
        T[:,2,2] = 1
        T[:,0,2] = -c[:,0]*scale
        T[:,1,2] = -c[:,1]*scale

        pts_out = torch.bmm(T, pts.permute(0,2,1))

        return pts_out, T

    def weighted_svd(self, pts1, pts2, weights, if_print=False):
        """ main function: get fundamental matrix and residual
        params: 
            pts1 -> [B, N, 2]: first set of points
            pts2 -> [B, N, 2]: second set of points
            weights -> [B, N, 1]: predicted weights
        return:
            out -> [B, 3, 3]: F matrix
            residual -> [B, N, 1]: residual of the minimization function
        """
        device = weights.device
        weights = weights.squeeze(1).unsqueeze(2)

        ones = torch.ones_like(weights)
        if self.is_cuda:
            ones = ones.cuda()
        ## normalize the points
        pts1n, T1 = self.normalize(pts1, ones)
        pts2n, T2 = self.normalize(pts2, ones)

        p = torch.cat((pts2n[:,0].unsqueeze(1)*pts1n,
                       pts2n[:,1].unsqueeze(1)*pts1n,
                       pts1n), 1).permute(0,2,1)

        if self.normalize_SVD:
            p = torch.nn.functional.normalize(p, dim=2)
        X = p*weights

        out_b = []
        F_vecs_list = []
        ## usually use GPU to calculate SVD and F matrix
        if self.if_cpu_svd:
            X = set_nan2zero(X)  # check if NAN
            for b in range(X.size(0)):
                # logging.info(f"X[b]: {X[b]}")
                _, _, V = torch.svd(X[b].cpu())
                F = V[:,-1].view(3,3)
                F_vecs_list.append(V[:,-1]/(V[:,-1].norm()))
                U, S, V = torch.svd(F)
                F_ = U.mm((S*self.mask.cpu()).diag()).mm(V.t())
                out_b.append(F_.unsqueeze(0))
            out = torch.cat(out_b, 0)
            F_vecs= torch.stack(F_vecs_list)
        else:
            for b in range(X.size(0)):
                _, _, V = torch.svd(X[b])
                F = V[:,-1].view(3,3)
                F_vecs_list.append(V[:,-1]/(V[:,-1].norm()))
                U, S, V = torch.svd(F)
                F_ = U.mm((S*self.mask.to(device)).diag()).mm(V.t())
                out_b.append(F_.unsqueeze(0))
            out = torch.cat(out_b, 0)
            F_vecs = torch.stack(F_vecs_list)

        if if_print:
            print(F_vecs.size(), p.size(), weights.size())
            print('----F_vecs')
            print(F_vecs[0].detach().cpu().numpy())
            print('----p')
            print(p[0].detach().cpu().numpy())
            print('----weights')
            print(weights[:2].squeeze().detach().cpu().numpy(), torch.sum(weights[:2], dim=1).squeeze().detach().cpu().numpy())

        residual = (X @ F_vecs.unsqueeze(-1)).squeeze(-1) # [B, N, 1]

        out = T2.permute(0,2,1).bmm(out).bmm(T1)
        return out, residual.squeeze(-1)

    def get_unique(self, xs, topk, matches_good_unique_nums, pts1, pts2): # [B, N]
        xs_topk_list = []
        topK_indices_list = []
        pts1_list = []
        pts2_list = []

        for x, matches_good_unique_num, pt1, pt2 in zip(xs, matches_good_unique_nums, pts1, pts2):
            # x_unique = torch.unique(x) # no gradients!!!
            x_unique = x[:, :matches_good_unique_num]
            # print(x_unique_topK)
            x_unique_topK, topK_indices = torch.topk(x_unique, topk, dim=1)
            xs_topk_list.append(x_unique_topK)
            topK_indices_list.append(topK_indices.squeeze())

            pt1_topK, pt2_topK = pt1[topK_indices.squeeze(), :], pt2[topK_indices.squeeze(), :]
            pts1_list.append(pt1_topK)
            pts2_list.append(pt2_topK)
        return torch.stack(xs_topk_list), torch.stack(topK_indices_list), torch.stack(pts1_list), torch.stack(pts2_list)

    def forward(self, pts1, pts2, weights, if_print=False, matches_good_unique_num=None):
        out, residual = self.weighted_svd(pts1, pts2, weights, if_print=if_print)
        return out, residual


class DeepFNet():
    def __init__(self, net_1, net_2):
        # models
        self.net_1 = net_1
        self.net_2 = net_2
        # parameters
        self.fit = Fit(False, False, True, True)
        self.depth = 5

    def predict(self, pts_normalized_in, pts1, pts2, T1, T2, matches_good_unique_num, t_scene_scale):
        pts_normalized_in = pts_normalized_in.to('cpu').detach().numpy().copy()
        pts1 = pts1.to('cpu').detach().numpy().copy()
        pts2 = pts2.to('cpu').detach().numpy().copy()
        T1 = T1.to('cpu').detach().numpy().copy()
        T2 = T2.to('cpu').detach().numpy().copy()
        matches_good_unique_num = np.array([matches_good_unique_num])
        #matches_good_unique_num = matches_good_unique_num.to('cpu').detach().numpy().copy()
        #t_scene_scale = t_scene_scale.to('cpu').detach().numpy().copy()
        
        logits = self.net_1.predict([pts_normalized_in])[0]

        logits = torch.from_numpy(logits.astype(np.float32)).clone()
        pts_normalized_in = torch.from_numpy(pts_normalized_in.astype(np.float32)).clone()
        pts1 = torch.from_numpy(pts1.astype(np.float32)).clone()
        pts2 = torch.from_numpy(pts2.astype(np.float32)).clone()
        T1 = torch.from_numpy(T1.astype(np.float32)).clone()
        T2 = torch.from_numpy(T2.astype(np.float32)).clone()
        matches_good_unique_num = torch.from_numpy(matches_good_unique_num.astype(np.float32)).clone()
        t_scene_scale = torch.from_numpy(t_scene_scale.astype(np.float32)).clone()

        weights_pts = F.softmax(logits, dim=2)
        weights_prod = weights_pts

        out_layers = []
        epi_res_layers = []
        residual_layers = []
        weights_layers = [weights_prod]
        logits_layers = [logits]

        ## recurrent network for updated weights
        for iter in range(self.depth-1):
            ## calculate residual using current weights
            out, residual = self.fit(pts1, pts2, weights_prod, matches_good_unique_num=matches_good_unique_num)

            out_layers.append(out)
            residual_layers.append(residual)
            epi_res = compute_epi_residual(pts1, pts2, out).unsqueeze(1)
            epi_res_layers.append(epi_res)

            ## combine the input, output, and residuals for the next run
            net_in = torch.cat((pts_normalized_in, weights_prod, epi_res, residual.unsqueeze(1)), 1)

            net_in = net_in.to('cpu').detach().numpy().copy()
            logits = self.net_2.predict(net_in)
            logits = torch.from_numpy(logits.astype(np.float32)).clone()

            weights_pts = F.softmax(logits, dim=2)
            weights_prod = weights_pts

            ## add intermediate output
            weights_layers.append(weights_prod)
            logits_layers.append(logits)

        ## last run of residual
        out, residual = self.fit(pts1, pts2, weights_prod, if_print=False, matches_good_unique_num=matches_good_unique_num)
        residual_layers.append(residual)
        out_layers.append(out)

        return logits.squeeze(1), logits_layers, out, epi_res_layers, T1, T2, out_layers, pts1, pts2, weights_prod, residual_layers, weights_layers


### Visualize ###
def rot12_to_angle_error(R0, R1):
    r, _ = cv2.Rodrigues(R0.dot(R1.T))
    rotation_error_from_identity = np.linalg.norm(r) / np.pi * 180.
    # another_way = np.rad2deg(np.arccos(np.clip((np.trace(R0 @ (R1.T)) - 1) / 2, -1., 1.)))
    # print(rotation_error_from_identity, another_way)
    return rotation_error_from_identity
    # return another_way

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v)) + 1e-10

def vector_angle(v1, v2):
    """ v1, v2: [N, 1] or (N)
        return: angles in degree: () """
    dot_product = dotproduct(v1, v2) / (length(v1) * length(v2) + 1e-10)
    return math.acos(np.clip(dot_product, -1., 1.)) / np.pi * 180.

def _homo(x):
    # input: x [N, 2] or [batch_size, N, 2]
    # output: x_homo [N, 3]  or [batch_size, N, 3]
    assert len(x.size()) in [2, 3]
    print(f"x: {x.size()[0]}, {x.size()[1]}, {x.dtype}, {x.device}")
    if len(x.size())==2:
        ones = torch.ones(x.size()[0], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 1)
    elif len(x.size())==3:
        ones = torch.ones(x.size()[0], x.size()[1], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 2)
    return x_homo

def _de_homo(x_homo):
    # input: x_homo [N, 3] or [batch_size, N, 3]
    # output: x [N, 2] or [batch_size, N, 2]
    assert len(x_homo.size()) in [2, 3]
    epi = 1e-10
    if len(x_homo.size())==2:
        x = x_homo[:, :-1]/((x_homo[:, -1]+epi).unsqueeze(-1))
    else:
        x = x_homo[:, :, :-1]/((x_homo[:, :, -1]+epi).unsqueeze(-1))
    return x

def compute_epi_residual(pts1, pts2, F, clamp_at=0.5):
    import logging
    l1 = pts2 @ F
    l2 = pts1 @ F.permute(0,2,1)

    dd = ((pts1*l1).sum(2))
    epi = 1e-6
    d = dd.abs()*(1/(l1[:,:,:2].norm(2,2) + epi) + 1/(l2[:,:,:2].norm(2,2) + epi)) # d1+d2, no squared
    # logging.info(f'd: {d}, dd: {dd}')
    # dd = d.pow(2)

    out = torch.clamp(d, max=clamp_at)

    return out

def draw_corr(im1, im2, x1, x2, linewidth=2, 
                new_figure=True, title='', color='g', if_show=True,
                zorder=1):
    # im1 = img1_rgb
    # im2 = img2_rgb
    # x1 = x1_sample
    # x2 = x2_sample
    im_shape = im1.shape
    assert im1.shape == im2.shape, 'Shape mismatch between im1 and im2! @draw_corr()'
    x2_copy = x2.copy()
    x2_copy[:, 0] = x2_copy[:, 0] + im_shape[1]
    im12 = np.hstack((im1, im2))

    if new_figure: 
        plt.figure(figsize=(60, 8))
    plt.imshow(im12, cmap=None if len(im12.shape)==3 else plt.get_cmap('gray'))
    plt.plot(np.vstack((x1[:, 0], x2_copy[:, 0])), np.vstack((x1[:, 1], x2_copy[:, 1])), 
                marker='', linewidth=linewidth, color=color, zorder=zorder)
    if title!='':
        plt.title(title)

def scatter_xy(xy, c, im_shape, title='', new_figure=True, s=2, 
                cmap='rainbow', set_lim=True, if_show=True, zorder=2):
    if new_figure:
        plt.figure(figsize=(60, 8))
    # plt.scatter(xy[:, 0], xy[:, 1], s=s, c=c, marker='o', cmap=cmap, zorder=zorder)
    plt.scatter(xy[:, 0], xy[:, 1], s=s, facecolors='none', linewidth=4, edgecolors=c, marker='o', cmap=cmap, zorder=zorder)
    # plt.scatter(xy[:, 0], xy[:, 1], s=s, c=c, marker='o', cmap=cmap, zorder=zorder)
    # plt.colorbar()
    if set_lim:
        plt.xlim(0, im_shape[1]-1)
        plt.ylim(im_shape[0]-1, 0)
    plt.title(title)
    val_inds = within(xy[:, 0], xy[:, 1], im_shape[1], im_shape[0])
    return val_inds

def within(x, y, xlim, ylim):
    val_inds = (x >= 0) & (y >= 0)
    val_inds = val_inds & (x <= xlim) & (y <= ylim)
    return val_inds

def goodCorr_eval_nondecompose(p1s, p2s, E_hat, delta_Rtij_inv, K, scores, if_my_decomp=False):
    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    if scores is not None:
        num_top = len(scores) // 10
        num_top = max(1, num_top)
        th = np.sort(scores)[::-1][num_top] ## [RUI] Only evaluating the top 10% corres.
        mask = scores >= th

        p1s_good = p1s[mask]
        p2s_good = p2s[mask]
    else:
        p1s_good, p2s_good = p1s, p2s

    # Match types
    # E_hat = E_hat.reshape(3, 3).astype(p1s.dtype))
    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        # num_inlier, R, t, mask_new = cv2.recoverPose(
        #     E_hat, p1s_good, p2s_good)
        if if_my_decomp:
            M2_list, error_Rt, Rt_cam = _E_to_M(torch.from_numpy(E_hat), torch.from_numpy(p1s_good), torch.from_numpy(p2s_good), delta_Rt_gt=delta_Rtij_inv, show_debug=False, method_name='Ours_best%d'%best_N)
            if not Rt_cam:
                return None, None
            else:
                print(Rt_cam[0], Rt_cam[1])
        else:
            num_inlier, R, t, mask_new = cv2.recoverPose(E_hat, p1s_good, p2s_good, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))
        try:
            R_cam, t_cam = utils_geo.invert_Rt(R, t)
            err_q = utils_geo.rot12_to_angle_error(R_cam, delta_Rtij_inv[:3, :3])
            err_t = utils_geo.vector_angle(t_cam, delta_Rtij_inv[:3, 3:4])
            # err_q, err_t = evaluate_R_t(dR, dt, R, t) # (3, 3) (3,) (3, 3) (3, 1)
        except:
            print("Failed in evaluation")
            print(R)
            print(t)
            err_q = 180.
            err_t = 90.
    else:
        err_q = 180.
        err_t = 90.
        R = np.eye(3).astype(np.float32)
        t = np.zeros((3, 1), np.float32)

    return np.hstack((R, t)), (err_q, err_t)

def epi_distance_np(F, X, Y, if_homo=False):
    # Not squared. https://arxiv.org/pdf/1706.07886.pdf
    if not if_homo:
        X = homo_np(X)
        Y = homo_np(Y)
    if len(X.shape)==2:
        nominator = np.abs(np.diag(Y@F@X.T))
        Fx1 = F @ X.T
        Fx2 = F.T @ Y.T
        denom_recp_Y_to_FX = 1./np.sqrt(Fx1[0]**2 + Fx1[1]**2)
        denom_recp_X_to_FY = 1./np.sqrt(Fx2[0]**2 + Fx2[1]**2)
    else:
        nominator = np.abs(np.diagonal(np.transpose(Y@F@X, (1, 2)), axis=1, axis2=2))
        Fx1 = F @np.transpose(X, (1, 2))
        Fx2 = np.transpose(F, (1, 2)) @ np.transpose(Y, (1, 2))
        denom_recp_Y_to_FX = 1./np.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2)
        denom_recp_X_to_FY = 1./np.sqrt(Fx2[:, 0]**2 + Fx2[:, 1]**2)
        # print(nominator.size(), denom.size())
    dist1 = nominator * denom_recp_Y_to_FX
    dist2 = nominator * denom_recp_X_to_FY
    dist3 = nominator * (denom_recp_Y_to_FX + denom_recp_X_to_FY)
    # return (dist1+dist2)/2., dist1, dist2
    return dist3, dist1, dist2

def homo_np(x):
    # input: x [N, D]
    # output: x_homo [N, D+1]
    N = x.shape[0]
    x_homo = np.hstack((x, np.ones((N, 1), dtype=x.dtype)))
    return x_homo

def Rt_pad(Rt):
    # Padding 3*4 [R|t] to 4*4 [[R|t], [0, 1]]
    assert Rt.shape==(3, 4)
    return np.vstack((Rt, np.array([[0., 0., 0., 1.]], dtype=Rt.dtype)))

def invert_Rt(R21, t21):
    delta_Rtij = Rt_depad(np.linalg.inv(Rt_pad(np.hstack((R21, t21)))))
    R12 = delta_Rtij[:, :3]
    t12 = delta_Rtij[:, 3:4]
    return R12, t12

def Rt_depad(Rt01):
    # dePadding 4*4 [[R|t], [0, 1]] to 3*4 [R|t]
    assert Rt01.shape==(4, 4)
    return Rt01[:3, :]

def rot12_to_angle_error(R0, R1):
    r, _ = cv2.Rodrigues(R0.dot(R1.T))
    rotation_error_from_identity = np.linalg.norm(r) / np.pi * 180.
    # another_way = np.rad2deg(np.arccos(np.clip((np.trace(R0 @ (R1.T)) - 1) / 2, -1., 1.)))
    # print(rotation_error_from_identity, another_way)
    return rotation_error_from_identity
    # return another_way

def vector_angle(v1, v2):
    """ v1, v2: [N, 1] or (N)
        return: angles in degree: () """
    dot_product = dotproduct(v1, v2) / (length(v1) * length(v2) + 1e-10)
    return math.acos(np.clip(dot_product, -1., 1.)) / np.pi * 180.

def recover_camera_opencv(K, x1, x2, delta_Rtij_inv, five_point=False, threshold=0.1, show_result=True, c=False, \
    if_normalized=False, method_app='', E_given=None, RANSAC=True):
    # Computes scene motion from x1 to x2
    # Compare with OpenCV with refs from:
    ## https://github.com/vcg-uvic/learned-correspondence-release/blob/16bef8a0293c042c0bd42f067d7597b8e84ef51a/tests.py#L232
    ## https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    ## http://answers.opencv.org/question/90070/findessentialmat-or-decomposeessentialmat-do-not-work-correctly/
    method_name = '5 point'+method_app if five_point else '8 point'+method_app
    if RANSAC:
        sample_method = cv2.RANSAC
    else:
        sample_method = None

    if show_result:
        print('>>>>>>>>>>>>>>>> Running OpenCV camera pose estimation... [%s] ---------------'%method_name)

    # Mostly following: # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    if E_given is None:
        if five_point:
            if if_normalized:
                E_5point, mask1 = cv2.findEssentialMat(x1, x2, method=sample_method, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
            else:
                E_5point, mask1 = cv2.findEssentialMat(x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=sample_method, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
            # x1_norm = cv2.undistortPoints(np.expand_dims(x1, axis=1), cameraMatrix=K, distCoeffs=None) 
            # x2_norm = cv2.undistortPoints(np.expand_dims(x2, axis=1), cameraMatrix=K, distCoeffs=None)
            # E_5point, mask = cv2.findEssentialMat(x1_norm, x2_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=threshold) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
        else:
            # F_8point, mask1 = cv2.findFundamentalMat(x1, x2, method=cv2.RANSAC) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
            F_8point, mask1 = cv2.findFundamentalMat(x1, x2, cv2.RANSAC, 0.1) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
            E_8point = K.T @ F_8point @ K
            U,S,V = np.linalg.svd(E_8point)
            E_8point = U @ np.diag([1., 1., 0.]) @ V
            # mask1 = np.ones((x1.shape[0], 1), dtype=np.uint8)
            print('8 pppppoint!')

        E_recover = E_5point if five_point else E_8point
    else:
        E_recover = E_given
        print('Use given E @recover_camera_opencv')
        mask1 = np.ones((x1.shape[0], 1), dtype=np.uint8)

    if if_normalized:
        if E_given is None:
            points, R, t, mask2 = cv2.recoverPose(E_recover, x1, x2, mask=mask1.copy()) # returns the inliers (subset of corres that pass the Cheirality check)
        else:
            points, R, t, mask2 = cv2.recoverPose(E_recover.astype(np.float64), x1, x2) # returns the inliers (subset of corres that pass the Cheirality check)
    else:
        if E_given is None:
            points, R, t, mask2 = cv2.recoverPose(E_recover, x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]), mask=mask1.copy())
        else:
            points, R, t, mask2 = cv2.recoverPose(E_recover.astype(np.float64), x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))

    # print(R, t)
    # else:
        # points, R, t, mask = cv2.recoverPose(E_recover, x1, x2)
    if show_result:
        print('# (%d, %d)/%d inliers from OpenCV.'%(np.sum(mask1!=0), np.sum(mask2!=0), mask2.shape[0]))

    R_cam, t_cam = invert_Rt(R, t)

    error_R = rot12_to_angle_error(R_cam, delta_Rtij_inv[:3, :3])
    error_t = vector_angle(t_cam, delta_Rtij_inv[:3, 3:4])
    if show_result:
        print('Recovered by OpenCV %s (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f'%(method_name, error_R, error_t))
        print(np.hstack((R, t)))

    # M_r = np.hstack((R, t))
    # M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    # P_l = np.dot(K,  M_l)
    # P_r = np.dot(K,  M_r)
    # point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1))
    # point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    # point_3d = point_4d[:3, :].T
    # scipy.io.savemat('test.mat', {'X': point_3d})

    if show_result:
        print('<<<<<<<<<<<<<<<< DONE Running OpenCV camera pose estimation. ---------------')

    E_return  = E_recover if five_point else (E_recover, F_8point)
    return np.hstack((R, t)), (error_R, error_t), mask2.flatten()>0, E_return

def epi_distance_np(F, X, Y, if_homo=False):
    # Not squared. https://arxiv.org/pdf/1706.07886.pdf
    if not if_homo:
        X = homo_np(X)
        Y = homo_np(Y)
    if len(X.shape)==2:
        nominator = np.abs(np.diag(Y@F@X.T))
        Fx1 = F @ X.T
        Fx2 = F.T @ Y.T
        denom_recp_Y_to_FX = 1./np.sqrt(Fx1[0]**2 + Fx1[1]**2)
        denom_recp_X_to_FY = 1./np.sqrt(Fx2[0]**2 + Fx2[1]**2)
    else:
        nominator = np.abs(np.diagonal(np.transpose(Y@F@X, (1, 2)), axis=1, axis2=2))
        Fx1 = F @np.transpose(X, (1, 2))
        Fx2 = np.transpose(F, (1, 2)) @ np.transpose(Y, (1, 2))
        denom_recp_Y_to_FX = 1./np.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2)
        denom_recp_X_to_FY = 1./np.sqrt(Fx2[:, 0]**2 + Fx2[:, 1]**2)
        # print(nominator.size(), denom.size())
    dist1 = nominator * denom_recp_Y_to_FX
    dist2 = nominator * denom_recp_X_to_FY
    dist3 = nominator * (denom_recp_Y_to_FX + denom_recp_X_to_FY)
    # return (dist1+dist2)/2., dist1, dist2
    return dist3, dist1, dist2

def show_epipolar_rui_gtEst(x1, x2, img1_rgb, img2_rgb, F_gt, F_est, im_shape, title_append='', 
                        emphasis_idx=[], label_text=False, weights=None, if_show=True,
                        linewidth=1.0):
    N_points = x1.shape[0]
    x1_homo = homo_np(x1)
    x2_homo = homo_np(x2)
    right_P = np.matmul(F_gt, x1_homo.T)
    right_epipolar_x = np.tile(np.array([[0], [1]]), N_points) * im_shape[1]
    # Using the eqn of line: ax+by+c=0; y = (-c-ax)/b, http://ai.stanford.edu/~mitul/cs223b/draw_epipolar.m
    right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]

    # colors = get_spaced_colors(x2.shape[0])
    # colors = np.random.random((x2.shape[0], 3))
    # plt.figure(figsize=(60, 8))
    plt.figure(figsize=(30, 4))
    plt.imshow(img2_rgb, cmap=None if len(img2_rgb.shape)==3 else plt.get_cmap('gray'))

    plt.plot(right_epipolar_x, right_epipolar_y, 'b', linewidth=linewidth, zorder=1)
    if weights is None:
        print(f"x2: {x2.shape}")
        plt.scatter(x2[:, 0], x2[:, 1], s=50, c='r', edgecolors='w', zorder=2)
    else:
        plt.scatter(x2[:, 0], x2[:, 1], s=weights*10000, c='r', edgecolors='w', zorder=2)

    if emphasis_idx:
        for idx in emphasis_idx:
            plt.scatter(x2[idx, 0], x2[idx, 1], s=80, color='y', edgecolors='w')



    if label_text:
        for idx in range(N_points):
            plt.text(x2[idx, 0], x2[idx, 1]-10, str(idx), fontsize=20, fontweight='extra bold', color='w')

    right_P = np.matmul(F_est, x1_homo.T)
    right_epipolar_x = np.tile(np.array([[0], [1]]), N_points) * im_shape[1]
    right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]
    plt.plot(right_epipolar_x, right_epipolar_y, 'g', linewidth=linewidth, zorder=1) # 'r'

    plt.xlim(0, im_shape[1]-1)
    plt.ylim(im_shape[0]-1, 0)
    # plt.title('Blue lines for GT F; Red lines for est. F. -- '+title_append)
    plt.title(f"{title_append}")

class Visualizer():
    def __init__(self, if_SP=False):
        self.if_SP = if_SP

    def get_img_data(self, imgs, idx=0):
        # Tests and vis
        #print(imgs)
        idx = 0
        img1 = imgs[0].astype(np.uint8)
        img2 = imgs[1].astype(np.uint8)
        img1_rgb, img2_rgb = img1, img2
        return {"img1_rgb": img1_rgb, "img2_rgb": img2_rgb}
        pass

    def val_rt(
        self,
        idx,
        K_np,
        x1_single_np,
        x2_single_np,
        E_est_np,
        E_gt_np,
        F_est_np,
        F_gt_np,
        delta_Rtijs_4_4_cpu_np,
        five_point,
        if_opencv=True,
    ):
        """ from essential matrix, get Rt, and error
        params:
            K_np:
            x1_single_np: matching point x1
            x2_single_np: matching point x2
            E_est_np, E_gt_np: essential matrix
            F_est_np, F_gt_np: fundamental matrix
            delta_Rtijs_4_4_cpu_np: ground truth transformation matrix
            five_point: with five_point or not (default not)
            if_opencv: compare to the results using opencv
        return:
            (pose error), (epipolar distance), (reconstructed poses)
        """
        delta_Rtij_inv = np.linalg.inv(delta_Rtijs_4_4_cpu_np)[:3]

        error_Rt_estW = None
        epi_dist_mean_estW = None
        error_Rt_opencv = None
        epi_dist_mean_opencv = None

        # Evaluating with our weights
        # _, error_Rt_estW = utils_F._E_to_M(E_est.detach(), K, x1_single_np, x2_single_np, w>0.5, \
        #     delta_Rtij_inv, depth_thres=500., show_debug=False, show_result=False, method_name='Est ws')
        M_estW, error_Rt_estW = goodCorr_eval_nondecompose(
            x1_single_np,
            x2_single_np,
            E_est_np.astype(np.float64),
            delta_Rtij_inv,
            K_np,
            None,
        )
        M_gt, error_Rt_gt = goodCorr_eval_nondecompose(
            x1_single_np,
            x2_single_np,
            E_gt_np.astype(np.float64),
            delta_Rtij_inv,
            K_np,
            None,
        )
        epi_dist_mean_estW, _, _ = epi_distance_np(
            F_est_np, x1_single_np, x2_single_np, if_homo=False
        )
        epi_dist_mean_gt, _, _ = epi_distance_np(
            F_gt_np, x1_single_np, x2_single_np, if_homo=False
        )

        # print('-0', F_est_np, epi_dist_mean_estW)

        # Evaluating with OpenCV 5-point
        if if_opencv:
            M_opencv, error_Rt_opencv, _, E_return = recover_camera_opencv(
                K_np,
                x1_single_np,
                x2_single_np,
                delta_Rtij_inv,
                five_point=five_point,
                threshold=0.01,
                show_result=False,
            )
            if five_point:
                E_recover_opencv = E_return
                F_recover_opencv = utils_F.E_to_F_np(E_recover_opencv, K_np)
            else:
                E_recover_opencv, F_recover_opencv = E_return[0], E_return[1]
            # print('+++', K_np)
            epi_dist_mean_opencv, _, _ = epi_distance_np(
                F_recover_opencv, x1_single_np, x2_single_np, if_homo=False
            )
            # print('-0-', utils_F.E_to_F_np(E_recover_5point, K_np))
            # print('-1', utils_F.E_to_F_np(E_recover_5point, K_np), epi_dist_mean_5point)
        return (
            error_Rt_estW, # error R,t
            epi_dist_mean_estW, # epipolar distance for each corr
            error_Rt_opencv,
            epi_dist_mean_opencv,
            error_Rt_gt, # for sanity check
            epi_dist_mean_gt, # epipolar distance wrt gt
            idx,
            M_estW, # reconstructed R, t
            M_opencv,
        )
        
    def get_val_rt(self, plot_data, idx=0, if_print=False):

        x1 = plot_data["x1"]
        x2 = plot_data["x2"]
        K = plot_data["K"]
        E_gt = plot_data["E_gt"]
        F_gt = plot_data["F_gt"]
        E_est = plot_data["E_est"]
        F_est = plot_data["F_est"]
        delta_Rtij = plot_data["delta_Rtij"]

        if if_print:
            print(f"F_gt: {F_gt/F_gt[2, 2]}")
            print(f"F_est: {F_est/F_est[2, 2]}")

        result = self.val_rt(
            idx,
            K,
            x1,
            x2,
            E_est,
            E_gt,
            F_est,
            F_gt,
            delta_Rtij,
            five_point=False,
            if_opencv=True,
        )
        error_Rt_estW, epi_dist_mean_estW, error_Rt_5point, epi_dist_mean_5point, error_Rt_gt, epi_dist_mean_gt = (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
        )

        if if_print:
            print(
                "Recovered by ours (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f"
                % (error_Rt_estW[0], error_Rt_estW[1])
            )
            #         print(epi_dist_mean_est_ours, np.mean(epi_dist_mean_est_ours))
            epi_dist_mean_est_ours = epi_dist_mean_estW
            print(
                "%.2f, %.2f"
                % (
                    np.sum(epi_dist_mean_est_ours < 0.1)
                    / epi_dist_mean_est_ours.shape[0],
                    np.sum(epi_dist_mean_est_ours < 1)
                    / epi_dist_mean_est_ours.shape[0],
                )
            )
        return {
            "error_Rt_est_ours": error_Rt_estW,
            "epi_dist_mean_est_ours": epi_dist_mean_estW,
            "epi_dist_mean_gt": epi_dist_mean_gt,
            # 'M_estW': M_estW,
        }
    
    def show(self, sample, out):
        imgs = sample["imgs"]                # [batch_size, H, W, 3]
        Ks = sample["K"]         # [batch_size, 3, 3]
        K_invs = sample["K_inv"] # [batch_size, 3, 3]
        #scene_names = sample["scene_name"]
        #frame_ids = sample["frame_ids"]
        scene_poses = sample["relative_scene_poses"]  # list of sequence_length tensors, which with size [batch_size, 4, 4]; the first being identity, the rest are [[R; t], [0, 1]]
        matches_all, matches_good = sample["matches_all"], sample["matches_good"]
        delta_Rtijs_4_4 = scene_poses[1]  # [batch_size, 4, 4], asserting we have 2 frames where scene_poses[0] are all identities
        E_gts, F_gts = sample["E"], sample["F"]
        pts1_virt_normalizedK, pts2_virt_normalizedK = sample["pts1_virt_normalized"], sample["pts2_virt_normalized"]
        pts1_virt_ori, pts2_virt_ori = sample["pts1_virt"], sample["pts2_virt"]

        # Show info from dataset
        delta_Rtijs_4_4_cpu_np = delta_Rtijs_4_4
        delta_Rtij_inv = np.linalg.inv(delta_Rtijs_4_4_cpu_np)
        angle_R = rot12_to_angle_error(np.eye(3), delta_Rtij_inv[:3, :3])
        angle_t = vector_angle(
            np.array([[0.0], [0.0], [1.0]]), delta_Rtij_inv[:3, 3:4]
        )
        print("Between frames: The rotation angle (degree) %.4f, and translation angle (degree) %.4f" % (angle_R, angle_t))

        # Get and Normalize points
        if self.if_SP:
            print('Not implemented')
            exit()
        else:
            matches_use = matches_good
            quality_use = None

        import torch
        # process x1, x2
        matches_use = matches_use[np.newaxis, :, :]
        N_corres = matches_use.shape[1]  # 1311 for matches_good, 2000 for matches_all
        inved_Ks = torch.inverse(torch.from_numpy(Ks))
        x1, x2 = (
            matches_use[:, :, :2],
            matches_use[:, :, 2:],
        )  # [batch_size, N, 2(W, H)]
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)
        x1_normalizedK = _de_homo(
            torch.matmul(inved_Ks, _homo(x1).transpose(1, 2)).transpose(1, 2)
        )  # [batch_size, N, 2(W, H)], min/max_X=[-W/2/f, W/2/f]
        x2_normalizedK = _de_homo(
            torch.matmul(inved_Ks, _homo(x2).transpose(1, 2)).transpose(1, 2)
        )  # [batch_size, N, 2(W, H)], min/max_X=[-W/2/f, W/2/f]
        matches_use_normalizedK = torch.cat((x1_normalizedK, x2_normalizedK), 2)
        matches_use_ori = torch.cat((x1, x2), 2)

        qs_scene = sample["q_scene"]  # [B, 4, 1]
        ts_scene = sample["t_scene"]  # [B, 3, 1]
        qs_cam = sample["q_cam"]  # [B, 4, 1]
        ts_cam = sample["t_cam"]  # [B, 3, 1]

        ts_scene = torch.from_numpy(ts_scene)
        t_scene_scale = torch.norm(ts_scene, p=2, dim=1, keepdim=True)

        data_batch = {
            "matches_xy_ori": matches_use_ori,
            "quality": quality_use,
            "x1_normalizedK": x1_normalizedK,
            "x2_normalizedK": x2_normalizedK,
            "Ks": Ks,
            "K_invs": K_invs,
            "matches_good_unique_nums": sample["matches_good_unique_nums"],
            "t_scene_scale": t_scene_scale,
            "pts1_virt_ori": pts1_virt_ori,
            "pts2_virt_ori": pts2_virt_ori,
        }

        loss_params = {
            "model": config["model"]["name"],
            "clamp_at": config["model"]["clamp_at"],
            "depth": config["model"]["depth"],
        }

        plot_data = {
            "Ks": Ks,
            "x1": x1,
            "x2": x2,
            "delta_Rtijs_4_4": delta_Rtijs_4_4,
            "F_gts": F_gts,
            "E_gts": E_gts,
        }

        print(pts1_virt_ori.shape, x1.shape, E_gts.shape, F_gts.shape)
        print(pts1_virt_ori[:10], x1[0][:10], E_gts, F_gts)
        
        # =======================================================
        # -> data_batch, loss_params, plot_data
        # =======================================================
        Ks = data_batch["Ks"]
        pts1_virt_ori, pts2_virt_ori = data_batch["pts1_virt_ori"], data_batch["pts2_virt_ori"]
        pts1_eval, pts2_eval = pts1_virt_ori, pts2_virt_ori

        logits_weights = out["weights"]
        loss_E = 0.0
        F_out, T1, T2, out_a = out["F_est"], out["T1"], out["T2"], out["out_layers"]

        pts1_virt_ori = torch.from_numpy(pts1_virt_ori).unsqueeze(0)
        pts2_virt_ori = torch.from_numpy(pts2_virt_ori).unsqueeze(0)
        pts1_eval = torch.bmm(T1, pts1_virt_ori.permute(0, 2, 1)).permute(0, 2, 1)
        pts2_eval = torch.bmm(T2, pts2_virt_ori.permute(0, 2, 1)).permute(0, 2, 1)

        loss_layers = []
        losses_layers = []
        out_a.append(F_out)
        loss_all = 0.0
        for iter in range(loss_params["depth"]):
            losses = compute_epi_residual(
                pts1_eval, pts2_eval, out_a[iter], loss_params["clamp_at"]
            )
            losses_layers.append(losses)
            loss = losses.mean()
            loss_layers.append(loss)
            loss_all += loss

        loss_all = loss_all / len(loss_layers)

        F_ests = T2.permute(0, 2, 1).bmm(F_out.bmm(T1))
        Ks = torch.from_numpy(Ks[np.newaxis, :, :])
        E_ests = Ks.transpose(1, 2) @ F_ests @ Ks

        last_losses = losses_layers[-1].detach().cpu().numpy()

        print(last_losses)
        print(np.amax(last_losses, axis=1))
        print(f"logits_weights: {logits_weights.shape}")
        outss =  {"E_ests": E_ests, "F_ests": F_ests, "logits_weights": logits_weights}

        # ここまで出力は正しそう？

        # =======================================================
        # data_batch, loss_params, plot_data → 
        # =======================================================
        print(sample.keys())
        #  TODO: ここから
        plot_data = plot_data
        d1 = plot_data
        print(plot_data.keys())
        plot_data.update(outss)
        print(plot_data.keys())

        # logging.info(f"plot_data: {plot_data}")

        # =======================================================
        # data = self.get_plot_data(plot_data, idx=0)
        # def get_plot_data(self, plot_data, idx=0, if_print=False):
        # =======================================================

        idx = 0
        ## convert all items to numpy
        plot_data_np = {}
        plot_data_np_idx = {}
        for i, en in enumerate(plot_data):
            if not isinstance(plot_data[en], np.ndarray): 
                plot_data_np[en] = plot_data[en].cpu().detach().numpy()
            else:
                plot_data_np[en] =  np.array(plot_data[en])

        name_map = {
            "Ks": "K",
            "x1": "x1",
            "x2": "x2",
            "E_ests": "E_est",
            "F_ests": "F_est",
            "F_gts": "F_gt",
            "E_gts": "E_gt",
            "delta_Rtijs_4_4": "delta_Rtij",
        }

        for i, en in enumerate(plot_data_np):
            name = name_map[en] if en in name_map else en
            print(en, plot_data_np[en].shape)
            if en in ['Ks', 'delta_Rtijs_4_4', 'F_gts', 'E_gts']:
                plot_data_np[en] = plot_data_np[en][np.newaxis, :, :]
            print(en, plot_data_np[en].shape)
            plot_data_np_idx[name] = plot_data_np[en][idx]
            print(plot_data_np_idx[name].shape)
        #exit()

        delta_Rtij = plot_data_np_idx["delta_Rtij"]

        print(delta_Rtij.shape)

        plot_data_np_idx["delta_Rtij_inv"] = np.linalg.inv(delta_Rtij)

        data = {"plot_data": plot_data_np_idx}

        # =======================================================
        # 
        # =======================================================

        plot_data = data["plot_data"]
        d2 = plot_data

        img_data = self.get_img_data(sample["imgs"])
        idx = sample['frame_ids']
        print('do plot_helper')

        # =======================================================
        # visualization
        # =======================================================
        #plot_helper(plot_data, img_data)
        """
        input:
            plot_data:
                dict: {'K', 'x1', 'x2', 'E_est', 'F_est', 
                       'F_gt', 'E_gt', 'delta_Rtij',
                       'logits_weights', }
            img_data:
                {'img1_rgb', 'img2_rgb'}
        """
        # plot epipolar lines.
        # plot correspondences
        img1_rgb = img_data["img1_rgb"]
        img2_rgb = img_data["img2_rgb"]
        x1 = plot_data["x1"]
        x2 = plot_data["x2"]

        if self.if_SP:
            x1_SP = plot_data["x1_SP"]
            x2_SP = plot_data["x2_SP"]
        K = plot_data["K"]
        E_gt = plot_data["E_gt"]
        F_gt = plot_data["F_gt"]
        E_est = plot_data["E_est"]
        F_est = plot_data["F_est"]
        scores_ori = plot_data["logits_weights"].flatten()
        # logging.info(f"scores_ori: {scores_ori.shape}")

        im_shape = img1_rgb.shape
        unique_rows_all, unique_rows_all_idxes = np.unique(
            np.hstack((x1, x2)), axis=0, return_index=True
        )

        def get_score_mask(scores_ori, unique_rows_all_idxes, num_corr=100, top=True):
            # num_corr = num_corr if top else -num_corr
            sort_idxes = np.argsort(scores_ori[unique_rows_all_idxes])[::-1]
            scores = scores_ori[unique_rows_all_idxes][sort_idxes]
            # num_corr = 100
            mask_conf = sort_idxes[:num_corr] if top else sort_idxes[-num_corr:]
            print(f" top 10: {scores[:10]}, last 10: {scores[-10:]}")
            return mask_conf

        def plot_corrs(
            unique_rows_all_idxes=None,
            mask_sample=None,
            title="corres.",
            savefile="test.png",
            axis_off=True
        ):
            def plot_scatter_xy(x1,img1_rgb,color='r',new_figure=False,zorder=2):
                unique_rows_all, unique_rows_all_idxes = np.unique(
                    x1, axis=0, return_index=True
                )
                scatter_xy(
                    unique_rows_all,
                    color,
                    img1_rgb.shape,
                    title="",
                    new_figure=new_figure,
                    s=100, # 100
                    set_lim=False,
                    if_show=False,
                    cmap=None,
                    zorder=zorder
                )
                pass  
            assert unique_rows_all_idxes is not None or mask_sample is not None
            # if unique_rows_all_idxes is None:
            #     x1 = x1[mask_sample]
            #     x2 = x2[mask_sample]
            # else:
            #     x1 = x1[unique_rows_all_idxes][mask_conf, :]
            #     x2 = x2[unique_rows_all_idxes][mask_conf, :]
            new_figure_corr = True
            if_corr = True
            color_t = 'g' # 'C0'
            if if_corr:
                draw_corr(
                    img1_rgb,
                    img2_rgb,
                    # x1 = x1[mask_sample] if unique_rows_all_idxes is None else x1[unique_rows_all_idxes][mask_conf, :],
                    # x2 = x2[mask_sample] if unique_rows_all_idxes is None else x2[unique_rows_all_idxes][mask_conf, :],
                    x1=x1[mask_sample]
                    if unique_rows_all_idxes is None
                    else x1[unique_rows_all_idxes],
                    x2=x2[mask_sample]
                    if unique_rows_all_idxes is None
                    else x2[unique_rows_all_idxes],
                    # x1[mask_sample],
                    # x2[mask_sample],
                    color=color_t,
                    new_figure=new_figure_corr,
                    linewidth=2,
                    title=title,
                    if_show=False,
                    zorder=1
                )
            if self.if_SP:
                print('Not implemented.')
                exit()

            # x2[:, 0] += img1_rgb.shape[1]
            x2_shift = x2 + 0
            x2_shift[:, 0] += img1_rgb.shape[1]
            plot_scatter_xy(x1,img1_rgb,color=color_t,new_figure=False,zorder=2)           
            plot_scatter_xy(x2_shift,img1_rgb,color=color_t,new_figure=False,zorder=2)           
            if axis_off:
                plt.axis('off')

            if savefile is not None:
                plt.savefig(savefile, dpi=300, bbox_inches="tight")

        def plot_epipolar(unique_rows_all_idxes, mask_conf, title="", savefile=None, axis_off=True):
            # if mask_conf is None:
            #     mask_conf = np.ones_like(unique_rows_all_idxes)

            # utils_vis.show_epipolar_rui_gtEst(
            #     x2[unique_rows_all_idxes][:],
            #     x1[unique_rows_all_idxes][:],
            #     img2_rgb,
            #     img1_rgb,
            #     F_gt.T,
            #     F_est.T,
            #     # weights=scores_ori[unique_rows_all_idxes],
            #     weights=None,
            #     im_shape=im_shape,
            #     title_append=title,
            #     if_show=False,
            #     linewidth=1.5
            # )

            show_epipolar_rui_gtEst(
                x2[unique_rows_all_idxes][mask_conf, :],
                x1[unique_rows_all_idxes][mask_conf, :],
                img2_rgb,
                img1_rgb,
                F_gt.T,
                F_est.T,
                weights=scores_ori[unique_rows_all_idxes][mask_conf],
                im_shape=im_shape,
                title_append=title,
                if_show=False,
                linewidth=1.5
            )
            if axis_off:
                plt.axis('off')
            if savefile is not None:
                plt.savefig(savefile, dpi=300, bbox_inches="tight")
            

        # base_folder = "plots/vis_paper"
        base_folder = "./"
        print(base_folder)
        #if "all" in plot_corr_list:
        if True:
            # for i, xs in enumerate(x1_SP):
            # print(f"x1_SP: {x1_SP.shape}")
            file = None
            file = f"{base_folder}/vis_corr_all.png"
            unique_rows_all, unique_rows_all_idxes = np.unique(
                x1, axis=0, return_index=True
            )
            plot_corrs(
                unique_rows_all_idxes=unique_rows_all_idxes,
                mask_sample=None,
                title=f"Sample of {unique_rows_all_idxes.shape[0]} corres.",
                savefile=file,
            )

        #if "random" in plot_corr_list:
        if True:
            # num = 100
            file = f"./vis_corr_all_random.png"
            unique_rows_all, unique_rows_all_idxes = np.unique(
                x1, axis=0, return_index=True
            )
            # logging.info(f"unique_rows_all: {unique_rows_all}, unique_rows_all: {unique_rows_all}")
            percentage = 0.3
            num = int(unique_rows_all.shape[0]*percentage)
            
            plot_corrs(
                mask_sample=unique_rows_all_idxes[np.random.choice(unique_rows_all_idxes.shape[0], num)],
                title=f"Sample of {num} corres.",
                savefile=file,
            )
        #if "mask_epi_dist_gt" in plot_corr_list:
        if True:
            num_points = 100
            data = self.get_val_rt(plot_data)
            mask_conf = get_score_mask(
                data["epi_dist_mean_gt"].flatten(),
                unique_rows_all_idxes,
                num_corr=num_points,
                top=False,
            )
            file = f"./vis_corr_all_mask_epi_dist_gt.png"
            plot_corrs(
                unique_rows_all_idxes,
                mask_conf,
                title=f"Top {mask_conf.shape[0]} correspondences with lowest epipolar distance", 
                savefile=file,
            )

        num_points = 80
        #if "mask_conf" in plot_epipolar_list:
        if True:
            file = f"./vis_mask_conf.png"
            print(f"scores_ori: {scores_ori.shape}, {scores_ori[0]}")
            mask_conf = get_score_mask(scores_ori, unique_rows_all_idxes, num_corr=num_points)
            print(f"mask_conf: {mask_conf}")

            ## sift version
            # print(f"x1: {x1.shape}, x2: {x2.shape}")
            # plot_epipolar(
            #     scores_ori,
            #     None,
            #     title=f"Ours top {mask_conf.shape[0]} with largest score points" if title else "",
            #     savefile=file
            # )

            # original
            plot_epipolar(
                unique_rows_all_idxes,
                mask_conf,
                title=f"Ours top {mask_conf.shape[0]} with largest score points",
                savefile=file
            )

        #if "mask_epi_dist_gt" in plot_epipolar_list:
        if True:
            data = self.get_val_rt(plot_data, if_print=True)
            # logging.info(f"data['epi_dist_mean_gt']: {data['epi_dist_mean_gt'].shape}")
            file = f"./vis_epi_dist_all.png"
            mask_conf = get_score_mask(
                data["epi_dist_mean_gt"].flatten(),
                unique_rows_all_idxes,
                num_corr=num_points,
                top=False,
            )
            plot_epipolar(
                unique_rows_all_idxes,
                mask_conf,
                title=f"Top {mask_conf.shape[0]} points with lowest epipolar distance",
                savefile=file
            )

        ## plot points selected from gt or deepF.
        pass


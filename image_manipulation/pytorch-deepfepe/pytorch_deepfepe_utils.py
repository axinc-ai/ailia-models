import argparse
import json
import os
import sys
import time

import ailia
import cv2
import numpy as np

from pytorch_deepfepe_utils import *

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
        "good_num": 1000
    }
}


# TODO: Eliminate pytorch dependency
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F


### Dataset ###
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

    def prepare(self, frame_id = '000000'):
        sequence_length = config["data"]["sequence_length"]
        delta_ij = config["data"]["delta_ij"]
        sizerHW = [600, 800] # omitted

        scene = frame[0]
        frame_id = frame[1]
        frame_num = int(frame_id)

        sample = {
            "scene": scene,
            "imgs": [],
            "ids": [],
            "relative_scene_poses": []
        }

        poses = self._load_as_array(os.path.join(scene, "poses.npy"))
        poses = poses.astype(np.float32).reshape(-1, 3, 4)

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
            sample["ids"].append(j)

        zoom_xys = []
        for img_file in sample['imgs']:
            _, zoom_xy = self._load_and_resize_img(img_file, sizerHW)
            zoom_xys.append(zoom_xy)
        zoom_xy = zoom_xys[-1] #### assume images are in the same size

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

        Rt_scene = sample["relative_scene_poses"][1]
        t_scene = Rt_scene[:3, 3:4]

        sample = {
            "matches_good": matches_good,
            "matches_good_unique_nums": matches_good_unique_nums,
            "t_scene": t_scene
        }

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

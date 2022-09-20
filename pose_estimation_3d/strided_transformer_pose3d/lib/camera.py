import sys
import numpy as np
import torch

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
	args = list(args)
	result = func(*args)
	return result

def qrot(q, v):
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv = np.cross(qvec, v, axis=len(q.shape) - 1)
	uuv = np.cross(qvec, uv, axis=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)


def get_uvd2xyz(uvd, gt_3D, cam):
    N, T, V,_ = uvd.size()

    dec_out_all = uvd.view(-1, T, V, 3).clone()  
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone() 

    cam_f_all = cam[..., :2].view(-1,1,1,2).repeat(1,T,V,1)
    cam_c_all = cam[..., 2:4].view(-1,1,1,2).repeat(1,T,V,1)

    z_global = dec_out_all[:, :, :, 2]
    z_global[:, :, 0] = root[:, :, 0, 2]
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]
    z_global = z_global.unsqueeze(-1) 
    
    uv = enc_in_all - cam_c_all 
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  
    xyz_global = torch.cat((xy, z_global), -1)
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))

    return xyz_offset


        
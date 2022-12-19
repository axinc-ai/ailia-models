import numpy as np


"""
    util/torch_util.py
"""
def expand_dim(tensor, dim, desired_dim_len):
    sz = list(tensor.shape)
    sz[dim] = desired_dim_len
    tensor = np.broadcast_to(tensor, tuple(sz))
    return tensor


"""
    geotnf/grid_gen.py
"""
class AffineGridGen():
    def __init__(self, out_h=240, out_w=240, out_ch = 3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def __call__(self, theta):
        b = theta.shape[0]
        if not theta.shape==(b,2,3):
            theta = theta.reshape(-1,2,3)
        batch_size = theta.shape[0]
        size = (batch_size, self.out_ch, self.out_h, self.out_w)

        #use_pytorch = True
        use_pytorch = False
        if not use_pytorch:
            # Patch of torch.nn.functional.affine_grid()
            # See below if you want to get original code.
            # https://github.com/pytorch/pytorch/issues/30563
            def affine_grid(theta, size, align_corners=False):
                N, C, H, W = size
                grid = create_grid(N, C, H, W)
                grid = grid.reshape(N, H * W, 3) @ theta.swapaxes(1, 2)
                grid = grid.reshape(N, H, W, 2)
                return grid

            def create_grid(N, C, H, W):
                grid = np.empty([N, H, W, 3], dtype='float32')
                # grid[..., 0] is equal to grid.select(-1, 0) in pytorch
                grid[..., 0] = np.copy(linspace_from_neg_one(W))
                grid[..., 1] = np.copy(np.expand_dims(linspace_from_neg_one(H), axis=-1))
                grid[..., 2] = np.full(grid[..., 2].shape, 1)
                return grid
                
            def linspace_from_neg_one(num_steps, dtype='float32'):
                r = np.linspace(-1, 1, num_steps, dtype=dtype)
                r = r * (num_steps - 1) / num_steps
                return r

            out = affine_grid(theta, size)
            
        else:
            import torch
            import torch.nn.functional as F

            theta = torch.tensor(theta)
            out_size = torch.Size(size)
            out = F.affine_grid(theta, out_size) # torch.nn.functional
            out = out.to('cpu').detach().numpy().copy()

        return out


class AffineGridGenV2():
    def __init__(self, out_h=240, out_w=240):
        super(AffineGridGenV2, self).__init__()
        self.out_h, self.out_w = out_h, out_w

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = self.grid_X[np.newaxis, :, :, np.newaxis]
        self.grid_Y = self.grid_Y[np.newaxis, :, :, np.newaxis]

    def __call__(self, theta):
        b = theta.shape[0]
        if not theta.shape==(b,6):
            theta = theta.reshape(b,6)

        t0=theta[:,0][:, np.newaxis, np.newaxis, np.newaxis]
        t1=theta[:,1][:, np.newaxis, np.newaxis, np.newaxis]
        t2=theta[:,2][:, np.newaxis, np.newaxis, np.newaxis]
        t3=theta[:,3][:, np.newaxis, np.newaxis, np.newaxis]
        t4=theta[:,4][:, np.newaxis, np.newaxis, np.newaxis]
        t5=theta[:,5][:, np.newaxis, np.newaxis, np.newaxis]

        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)
        grid_Xp = grid_X*t0 + grid_Y*t1 + t2
        grid_Yp = grid_X*t3 + grid_Y*t4 + t5

        return np.concatenate((grid_Xp, grid_Yp), axis=3)


class HomographyGridGen():
    def __init__(self, out_h=240, out_w=240):
        super(HomographyGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = self.grid_X[np.newaxis, :, :, np.newaxis]
        self.grid_Y = self.grid_Y[np.newaxis, :, :, np.newaxis]

    def __call__(self, theta):
        b = theta.shape[0]
        if theta.shape[1]==9:
            H = theta
        else:
            H = homography_mat_from_4_pts(theta)
        
        h0 = H[:,0][:, np.newaxis, np.newaxis, np.newaxis]
        h1 = H[:,1][:, np.newaxis, np.newaxis, np.newaxis]
        h2 = H[:,2][:, np.newaxis, np.newaxis, np.newaxis]
        h3 = H[:,3][:, np.newaxis, np.newaxis, np.newaxis]
        h4 = H[:,4][:, np.newaxis, np.newaxis, np.newaxis]
        h5 = H[:,5][:, np.newaxis, np.newaxis, np.newaxis]
        h6 = H[:,6][:, np.newaxis, np.newaxis, np.newaxis]
        h7 = H[:,7][:, np.newaxis, np.newaxis, np.newaxis]
        h8 = H[:,8][:, np.newaxis, np.newaxis, np.newaxis]

        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)

        grid_Xp = grid_X*h0+grid_Y*h1+h2
        grid_Yp = grid_X*h3+grid_Y*h4+h5
        k = grid_X*h6+grid_Y*h7+h8

        grid_Xp /= k; grid_Yp /= k

        out = np.concatenate((grid_Xp, grid_Yp), axis=3)

        return out


class TpsGridGen():
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=3, reg_factor=0):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w), np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = self.grid_X[np.newaxis, :, :, np.newaxis]
        self.grid_Y = self.grid_Y[np.newaxis, :, :, np.newaxis]

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)

            self.Li = self.compute_L_inverse(P_X, P_Y)[np.newaxis, :, :].astype(np.float32)
            self.P_X = P_X[:, :, np.newaxis, np.newaxis, np.newaxis].swapaxes(0, 4)
            self.P_Y = P_Y[:, :, np.newaxis, np.newaxis, np.newaxis].swapaxes(0, 4)

    def __call__(self, theta):
        points = np.concatenate((self.grid_X,self.grid_Y), axis=3)
        warped_grid = self.apply_transformation(theta, points)
        return warped_grid

    def compute_L_inverse(self,X,Y):
        N = X.shape[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = np.broadcast_to(X, (N, N)) # X.expand(N,N)
        Ymat = np.broadcast_to(Y, (N, N)) # Y.expand(N,N)

        Xmat = Xmat - np.swapaxes(Xmat, 0, 1)
        Ymat = Ymat - np.swapaxes(Ymat, 0, 1)

        P_dist_squared = np.power(Xmat, 2) + np.power(Ymat, 2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation

        K = np.multiply(P_dist_squared, np.log(P_dist_squared))
        if self.reg_factor != 0:
            K += np.eye(K.shape[0],K.shape[1]) * self.reg_factor
   
        # construct matrix L
        O = np.ones((N,1))
        Z = np.zeros((3,3))
        P = np.concatenate((O,X,Y), axis=1)

        L = np.concatenate((
            np.concatenate((K,P), axis=1),
            np.concatenate((P.swapaxes(0,1),Z), axis=1)
        ), axis=0)
        Li = np.linalg.inv(L)

        return Li

    def apply_transformation(self,theta,points):
        if len(theta.shape) == 2:
            theta = theta[:, :, np.newaxis, np.newaxis]

        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.shape[0]

        # split theta into point coordinates
        Q_X = theta[:,:self.N,:,:].squeeze(3)
        Q_Y = theta[:,self.N:,:,:].squeeze(3)
    
        # get spatial dimensions of points
        points_b = points.shape[0]
        points_h = points.shape[1]
        points_w = points.shape[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = np.broadcast_to(self.P_X, (1, points_h, points_w, 1, self.N)) # x.expand()
        P_Y = np.broadcast_to(self.P_Y, (1, points_h, points_w, 1, self.N)) # x.expand()

        # compute weigths for non-linear part
        W_X = np.broadcast_to(self.Li[:, :self.N, :self.N], (batch_size, self.N, self.N)) @ Q_X
        W_Y = np.broadcast_to(self.Li[:, :self.N, :self.N], (batch_size, self.N, self.N)) @ Q_Y

        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X[:, :, :, np.newaxis, np.newaxis]
        W_X = W_X.swapaxes(1, 4)
        W_X = np.tile(W_X, (1, points_h, points_w, 1, 1))
        W_Y = W_Y[:, :, :, np.newaxis, np.newaxis]
        W_Y = W_Y.swapaxes(1, 4)
        W_Y = np.tile(W_Y, (1, points_h, points_w, 1, 1))

        # compute weights for affine part
        A_X = np.broadcast_to(self.Li[:, self.N:, :self.N], (batch_size, 3, self.N)) @ Q_X
        A_Y = np.broadcast_to(self.Li[:, self.N:, :self.N], (batch_size, 3, self.N)) @ Q_Y

        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X[:, :, :, np.newaxis, np.newaxis]
        A_X = A_X.swapaxes(1, 4)
        A_X = np.tile(A_X, (1, points_h, points_w, 1, 1))
        A_Y = A_Y[:, :, :, np.newaxis, np.newaxis]
        A_Y = A_Y.swapaxes(1, 4)
        A_Y = np.tile(A_Y, (1, points_h, points_w, 1, 1))

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        p = points[:,:,:,0]
        p = p[:, :, :, np.newaxis, np.newaxis]
        points_X_for_summation = np.broadcast_to(p, points[:,:,:,0].shape + (1, self.N))
        p = points[:,:,:,1]
        p = p[:, :, :, np.newaxis, np.newaxis]
        points_Y_for_summation = np.broadcast_to(p, points[:,:,:,1].shape + (1, self.N))

        if points_b==1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            delta_X = points_X_for_summation - np.broadcast_to(P_X, points_X_for_summation.shape)
            delta_Y = points_Y_for_summation - np.broadcast_to(P_Y, points_Y_for_summation.shape)
            # TODO: check
            # use expanded P_X,P_Y in batch dimension
            #delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            #delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
    
        dist_squared = np.power(delta_X,2) + np.power(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0] = 1 # avoid NaN in log computation
        U = np.multiply(dist_squared, np.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0]
        points_X_batch = points_X_batch[:, :, :, np.newaxis]
        points_Y_batch = points[:,:,:,1]
        points_Y_batch = points_Y_batch[:, :, :, np.newaxis]

        if points_b==1:
            points_X_batch = np.broadcast_to(points_X_batch, (batch_size,) + points_X_batch.shape[1:])
            points_Y_batch = np.broadcast_to(points_Y_batch, (batch_size,) + points_Y_batch.shape[1:])

        points_X_prime = A_X[:,:,:,:,0]+ \
                       np.multiply(A_X[:,:,:,:,1], points_X_batch) + \
                       np.multiply(A_X[:,:,:,:,2], points_Y_batch) + \
                       np.sum(np.multiply(W_X, np.broadcast_to(U, W_X.shape)), axis=4)

        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       np.multiply(A_Y[:,:,:,:,1], points_X_batch) + \
                       np.multiply(A_Y[:,:,:,:,2], points_Y_batch) + \
                       np.sum(np.multiply(W_Y, np.broadcast_to(U, W_Y.shape)), axis=4)
        
        return np.concatenate((points_X_prime, points_Y_prime), axis=3)


"""
    geotnf/transformation.py
"""
def homography_mat_from_4_pts(theta):
    b = theta.shape[0]

    if not theta.shape == (b, 8):
        theta = theta.reshape(b, 8)
    
    xp = theta[:, :4]
    xp = xp[:, :, np.newaxis]
    yp = theta[:, 4:]
    yp = yp[:, :, np.newaxis]

    x = np.array([-1, -1, 1, 1])
    x = x[:, np.newaxis]
    x = x[np.newaxis, :, :]
    x = np.broadcast_to(x, (b, 4, 1)) # x.expand(b,4,1)

    y = np.array([-1, 1, -1, 1])
    y = y[:, np.newaxis]
    y = y[np.newaxis, :, :]
    y = np.broadcast_to(y, (b, 4, 1)) # y.expand(b,4,1)

    z = np.zeros(4)
    z = z[:, np.newaxis]
    z = z[np.newaxis, :, :]
    z = np.broadcast_to(z, (b, 4, 1)) # z.expand(b,4,1)

    o = np.ones(4)
    o = o[:, np.newaxis]
    o = o[np.newaxis, :, :]
    o = np.broadcast_to(o, (b, 4, 1)) # o.expand(b,4,1)

    single_o = np.ones(1)
    single_o = single_o[:, np.newaxis]
    single_o = single_o[np.newaxis, :, :]
    single_o = np.broadcast_to(single_o, (b, 1, 1)) # o.expand(b,1,1)

    A1 = np.concatenate((-x,-y,-o,z,z,z,x*xp,y*xp,xp), axis=2)
    A2 = np.concatenate((z,z,z,-x,-y,-o,x*yp,y*yp,yp), axis=2)
    A = np.concatenate((A1, A2), axis=1)

    h = np.linalg.inv(A[:, :, :8]) @ -A[:, :, 8][:, :, np.newaxis]
    h = np.concatenate((h, single_o), axis=1)
    H = h.squeeze(2)

    return H


"""
    geotnf/point_tnf.py
"""
class PointTnf(object):
    """

    Class with functions for transforming a set of points with affine/tps transformations

    """
    def __init__(self, tps_grid_size=3, tps_reg_factor=0):
        self.tpsTnf = TpsGridGen(grid_size=tps_grid_size, reg_factor=tps_reg_factor)

    def tpsPointTnf(self, theta, points):
        # points are expected in [B,2,N], where first row is X and second row is Y
        # reshape points for applying Tps transformation
        points = points[:, :, :, np.newaxis]
        points = np.swapaxes(points, 1, 3)
        # apply transformation
        warped_points = self.tpsTnf.apply_transformation(theta, points)
        # undo reshaping
        warped_points = warped_points.swapaxes(3, 1)
        warped_points = warped_points.squeeze(3)
        return warped_points

    """
    def homPointTnf(self,theta,points,eps=1e-5):
        b=theta.size(0)
        if theta.size(1)==9:
            H = theta
        else:
            # TODO: torch to numpy
            H = homography_mat_from_4_pts(theta)
        h0=H[:,0].unsqueeze(1).unsqueeze(2)
        h1=H[:,1].unsqueeze(1).unsqueeze(2)
        h2=H[:,2].unsqueeze(1).unsqueeze(2)
        h3=H[:,3].unsqueeze(1).unsqueeze(2)
        h4=H[:,4].unsqueeze(1).unsqueeze(2)
        h5=H[:,5].unsqueeze(1).unsqueeze(2)
        h6=H[:,6].unsqueeze(1).unsqueeze(2)
        h7=H[:,7].unsqueeze(1).unsqueeze(2)
        h8=H[:,8].unsqueeze(1).unsqueeze(2)

        X=points[:,0,:].unsqueeze(1)
        Y=points[:,1,:].unsqueeze(1)
        Xp = X*h0+Y*h1+h2
        Yp = X*h3+Y*h4+h5
        k = X*h6+Y*h7+h8
        # prevent division by 0
        k = k+torch.sign(k)*eps

        Xp /= k; Yp /= k

        return torch.cat((Xp,Yp),1)
    """

    """
    def affPointTnf(self,theta,points):
        theta_mat = theta.view(-1,2,3)
        warped_points = torch.bmm(theta_mat[:,:,:2],points)
        warped_points += theta_mat[:,:,2].unsqueeze(2).expand_as(warped_points)
        return warped_points
    """


def compose_H_matrices(H1,H2):
    H1 = H1.reshape(-1, 3, 3)
    H2 = H2.reshape(-1,3,3)
    H = H1 @ H2
    H = H.reshape(-1, 9)

    return H


def compose_aff_matrices(theta_1,theta_2):
    batch_size=theta_1.shape[0]
    O = np.zeros((batch_size, 1, 3))
    O[:, :, 2] = 1

    theta_1 = theta_1.reshape(-1, 2, 3)
    theta_2 = theta_2.reshape(-1, 2, 3)

    theta_1 = np.concatenate((theta_1, O), axis=1)
    theta_2 = np.concatenate((theta_2, O), axis=1)

    theta = theta_1 @ theta_2
    theta = theta[:, :2, :].reshape(batch_size, 6)

    return theta


def compose_tps(theta_1, theta_2):
    batch_size = theta_1.shape[0]
    pt = PointTnf()
    P_1_2 = pt.tpsPointTnf(theta=theta_1, points=theta_2.reshape(batch_size, 2, 9))
    theta = P_1_2.reshape(-1,18)

    return theta


"""
    geotnf/transformation.py
"""
class GeometricTnf(object):
    """

    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )

    """
    def __init__(self, geometric_model='affine', tps_grid_size=3, tps_reg_factor=0, out_h=240, out_w=240, offset_factor=None):
        self.out_h = out_h
        self.out_w = out_w
        self.geometric_model = geometric_model
        self.offset_factor = offset_factor

        print('geometric_model={}'.format(geometric_model))

        if geometric_model=='affine' and offset_factor is None:
            self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w)
        elif geometric_model=='affine' and offset_factor is not None:
            self.gridGen = AffineGridGenV2(out_h=out_h, out_w=out_w)
        elif geometric_model=='hom':
            self.gridGen = HomographyGridGen(out_h=out_h, out_w=out_w)
        elif geometric_model=='tps':
            self.gridGen = TpsGridGen(out_h=out_h, out_w=out_w, grid_size=tps_grid_size, reg_factor=tps_reg_factor)

        if offset_factor is not None:
            self.gridGen.grid_X = self.gridGen.grid_X / offset_factor
            self.gridGen.grid_Y = self.gridGen.grid_Y / offset_factor

        self.theta_identity = np.expand_dims(np.array([[1,0,0],[0,1,0]]), 0).astype(np.float32)

    def __call__(self, image_batch, theta_batch=None, out_h=None, out_w=None, return_warped_image=True, return_sampling_grid=False, padding_factor=1.0, crop_factor=1.0):
        if image_batch is None:
            b = 1
        else:
            b = image_batch.shape[0]
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = np.broadcast_to(theta_batch, (b, 2, 3))

        # check if output dimensions have been specified at call time and have changed
        if (out_h is not None and out_w is not None) and (out_h!=self.out_h or out_w!=self.out_w):
            if self.geometric_model=='affine':
                gridGen = AffineGridGen(out_h, out_w)
            elif self.geometric_model=='hom':
                gridGen = HomographyGridGen(out_h, out_w)
            elif self.geometric_model=='tps':
                gridGen = TpsGridGen(out_h, out_w)
        else:
            gridGen = self.gridGen

        sampling_grid = gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        if padding_factor != 1 or crop_factor !=1:
            sampling_grid = sampling_grid*(padding_factor*crop_factor)
        # rescale grid according to offset_factor
        if self.offset_factor is not None:
            sampling_grid = sampling_grid*self.offset_factor

        if return_sampling_grid and not return_warped_image:
            return sampling_grid
        else:
            # sample transformed image
            #warped_image_batch = F.grid_sample(image_batch, sampling_grid) # torch.nn.functional
            warped_image_batch = bilinear_grid_sample(image_batch, sampling_grid, align_corners=False)

            if return_sampling_grid and return_warped_image:
                return (warped_image_batch, sampling_grid)
            else:
                return warped_image_batch


def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.reshape(n, -1)
    y = y.reshape(n, -1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y))[:, np.newaxis, :]
    wb = ((x1 - x) * (y - y0))[:, np.newaxis, :]
    wc = ((x - x0) * (y1 - y))[:, np.newaxis, :]
    wd = ((x - x0) * (y - y0))[:, np.newaxis, :]

    # Apply default for grid_sample function zero padding
    #im_padded = torch.nn.functional.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    im_padded = np.pad(im, [(0, 0), (0, 0), (1, 1), (1, 1)], mode='constant')

    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = np.where(x0 < 0, 0, x0)
    x0 = np.where(x0 > padded_w - 1, padded_w - 1, x0)
    x1 = np.where(x1 < 0, 0, x1)
    x1 = np.where(x1 > padded_w - 1, padded_w - 1, x1)
    y0 = np.where(y0 < 0, 0, y0)
    y0 = np.where(y0 > padded_h - 1, padded_h - 1, y0)
    y1 = np.where(y1 < 0, 0, y1)
    y1 = np.where(y1 > padded_h - 1, padded_h - 1, y1)

    im_padded = im_padded.reshape(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w)[:, np.newaxis, :]
    x0_y1 = (x0 + y1 * padded_w)[:, np.newaxis, :]
    x1_y0 = (x1 + y0 * padded_w)[:, np.newaxis, :]
    x1_y1 = (x1 + y1 * padded_w)[:, np.newaxis, :]

    x0_y0 = np.tile(x0_y0, (1, c, 1))
    x0_y1 = np.tile(x0_y1, (1, c, 1))
    x1_y0 = np.tile(x1_y0, (1, c, 1))
    x1_y1 = np.tile(x1_y1, (1, c, 1))

    Ia = np.take_along_axis(im_padded, x0_y0, 2) # torch.gather
    Ib = np.take_along_axis(im_padded, x0_y1, 2) # torch.gather
    Ic = np.take_along_axis(im_padded, x1_y0, 2) # torch.gather
    Id = np.take_along_axis(im_padded, x1_y1, 2) # torch.gather

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

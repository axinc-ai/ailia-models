# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import abstractmethod

import numpy as np


def rotation_3d_in_axis(points, angles, axis=2, return_mat=False):
    """Rotate points by angles in axis.

    Args:
        points (np.array): Points to rotate, shape (..., 3)
        angles (float | np.array): Rotation angles
        axis (int): Rotation axis (0, 1, or 2 for x, y, z)
        return_mat (bool): Whether to return rotation matrix

    Returns:
        np.array: Rotated points
        np.array (optional): Rotation matrix if return_mat=True
    """
    if not isinstance(angles, np.ndarray):
        angles = np.array(angles)

    # Ensure angles is broadcastable with points
    if angles.ndim == 0:
        angles = angles.reshape(1)

    original_shape = points.shape
    if points.ndim == 3:
        # Reshape to (N*M, 3) for batch processing
        N, M, _ = points.shape
        points = points.reshape(-1, 3)
        if angles.shape[0] == N:
            angles = np.repeat(angles, M)

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Create rotation matrices
    if axis == 0:  # rotation around x-axis
        rot_mat = np.zeros((len(angles), 3, 3))
        rot_mat[:, 0, 0] = 1
        rot_mat[:, 1, 1] = cos_a
        rot_mat[:, 1, 2] = -sin_a
        rot_mat[:, 2, 1] = sin_a
        rot_mat[:, 2, 2] = cos_a
    elif axis == 1:  # rotation around y-axis
        rot_mat = np.zeros((len(angles), 3, 3))
        rot_mat[:, 0, 0] = cos_a
        rot_mat[:, 0, 2] = sin_a
        rot_mat[:, 1, 1] = 1
        rot_mat[:, 2, 0] = -sin_a
        rot_mat[:, 2, 2] = cos_a
    elif axis == 2:  # rotation around z-axis
        rot_mat = np.zeros((len(angles), 3, 3))
        rot_mat[:, 0, 0] = cos_a
        rot_mat[:, 0, 1] = -sin_a
        rot_mat[:, 1, 0] = sin_a
        rot_mat[:, 1, 1] = cos_a
        rot_mat[:, 2, 2] = 1
    else:
        raise ValueError(f"Invalid axis {axis}")

    # Apply rotation
    rotated_points = np.einsum("nij,nj->ni", rot_mat, points)

    # Reshape back to original shape
    if len(original_shape) == 3:
        rotated_points = rotated_points.reshape(original_shape)

    if return_mat:
        if len(angles) == 1:
            return rotated_points, rot_mat[0]
        else:
            return rotated_points, rot_mat
    else:
        return rotated_points


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (np.array): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period (float, optional): Period of the value. Defaults to np.pi.

    Returns:
        np.array: Value in the range of [offset * period, (offset + 1) * period)
    """
    return val - np.floor(val / period + offset) * period


def box_iou_rotated(boxes1, boxes2):
    """Calculate IoU of rotated boxes. This is a simplified implementation.

    Args:
        boxes1 (np.array): Boxes in format [x, y, w, h, angle]
        boxes2 (np.array): Boxes in format [x, y, w, h, angle]

    Returns:
        np.array: IoU matrix
    """
    # This is a simplified implementation - in practice you'd need a more
    # sophisticated rotated box IoU calculation
    # For now, return a simple approximation using axis-aligned boxes
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]

    # Create IoU matrix with zeros (placeholder)
    iou_matrix = np.zeros((len(boxes1), len(boxes2)))

    # This is a placeholder - implement proper rotated box IoU if needed
    return iou_matrix


def points_in_boxes_part(points, boxes):
    """Find the box in which each point is. Simplified implementation.

    Args:
        points (np.array): Points in shape (1, M, 3) or (M, 3)
        boxes (np.array): Boxes in shape (1, N, box_dim) or (N, box_dim)

    Returns:
        np.array: Box indices for each point
    """
    # Simplified implementation - returns -1 for all points
    if points.ndim == 3:
        return np.full(points.shape[1], -1, dtype=np.int64)
    else:
        return np.full(points.shape[0], -1, dtype=np.int64)


def points_in_boxes_all(points, boxes):
    """Find all boxes in which each point is. Simplified implementation.

    Args:
        points (np.array): Points in shape (1, M, 3) or (M, 3)
        boxes (np.array): Boxes in shape (1, N, box_dim) or (N, box_dim)

    Returns:
        np.array: Boolean matrix indicating point-box relationships
    """
    # Simplified implementation
    if points.ndim == 3:
        M = points.shape[1]
    else:
        M = points.shape[0]

    if boxes.ndim == 3:
        N = boxes.shape[1]
    else:
        N = boxes.shape[0]

    return np.zeros((M, N), dtype=bool)


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Defaults to True.
        origin (tuple[float], optional): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.size == 0:
            tensor = tensor.reshape((0, box_dim)).astype(np.float32)
        # assert tensor.ndim == 2 and tensor.shape[-1] == box_dim, tensor.shape

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = np.zeros((tensor.shape[0], 1), dtype=tensor.dtype)
            tensor = np.concatenate((tensor, fake_rot), axis=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.copy()

        if origin != (0.5, 0.5, 0):
            dst = np.array((0.5, 0.5, 0))
            src = np.array(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        """np.array: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """np.array: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """np.array: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def height(self):
        """np.array: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 5]

    @property
    def top_height(self):
        """np.array:
        A vector with the top height of each box in shape (N, )."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """np.array:
        A vector with bottom's height of each box in shape (N, )."""
        return self.tensor[:, 2]

    @property
    def center(self):
        """Calculate the center of all the boxes.

        Note:
            In MMDetection3D's convention, the bottom center is
            usually taken as the default center.

            The relative position of the centers in different kinds of
            boxes are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
            It is recommended to use ``bottom_center`` or ``gravity_center``
            for clearer usage.

        Returns:
            np.array: A tensor with center of each box in shape (N, 3).
        """
        return self.bottom_center

    @property
    def bottom_center(self):
        """np.array: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        """np.array: A tensor with center of each box in shape (N, 3)."""
        pass

    @property
    def corners(self):
        """np.array:
        a tensor with 8 corners of each box in shape (N, 8, 3)."""
        pass

    @property
    def bev(self):
        """np.array: 2D BEV box of each box with rotation
        in XYWHR format, in shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self):
        """np.array: A tensor of 2D BEV box of each box
        without rotation."""
        # Obtain BEV boxes with rotation in XYWHR format
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = np.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = np.where(
            conditions, bev_rotated_boxes[:, [0, 1, 3, 2]], bev_rotated_boxes[:, :4]
        )

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)
        return bev_boxes

    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, x_max, y_max)

        Note:
            The original implementation of SECOND checks whether boxes in
            a range by checking whether the points are in a convex
            polygon, we reduce the burden for simpler cases.

        Returns:
            torch.Tensor: Whether each box is inside the reference range.
        """
        in_range_flags = (
            (self.bev[:, 0] > box_range[0])
            & (self.bev[:, 1] > box_range[1])
            & (self.bev[:, 0] < box_range[2])
            & (self.bev[:, 1] < box_range[3])
        )
        return in_range_flags

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | numpy.ndarray |
                :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """
        pass

    @abstractmethod
    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction.

        Args:
            bev_direction (str, optional): Direction by which to flip.
                Can be chosen from 'horizontal' and 'vertical'.
                Defaults to 'horizontal'.
        """
        pass

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (np.array): Translation vector of size (1, 3).
        """
        if not isinstance(trans_vector, np.ndarray):
            trans_vector = np.array(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each box is
                inside the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 2] > box_range[2])
            & (self.tensor[:, 0] < box_range[3])
            & (self.tensor[:, 1] < box_range[4])
            & (self.tensor[:, 2] < box_range[5])
        )
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type
                in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor  # velocity

    def limit_yaw(self, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float, optional): The offset of the yaw. Defaults to 0.5.
            period (float, optional): The expected period. Defaults to np.pi.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold=0.0):
        """Find boxes that are non-empty.

        A box is considered empty,
        if either of its side is no larger than threshold.

        Args:
            threshold (float, optional): The threshold of minimal sizes.
                Defaults to 0.0.

        Returns:
            torch.Tensor: A binary vector which represents whether each
                box is empty (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = (size_x > threshold) & (size_y > threshold) & (size_z > threshold)
        return keep

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
                :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                box_dim=self.box_dim,
                with_yaw=self.with_yaw,
            )
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, boxes_list):
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (list[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated Boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(np.empty((0, 7)))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use np.concatenate
        # so the returned boxes never share storage with input
        cat_boxes = cls(
            np.concatenate([b.tensor for b in boxes_list], axis=0),
            box_dim=boxes_list[0].tensor.shape[1],
            with_yaw=boxes_list[0].with_yaw,
        )
        return cat_boxes

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties
                as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.copy(), box_dim=self.box_dim, with_yaw=self.with_yaw
        )

    @property
    def device(self):
        """str: The device of the boxes are on."""
        return "cpu"  # NumPy arrays are always on CPU

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A box of shape (4,).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode="iou"):
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between boxes1 and
            boxes2,  boxes1 and boxes2 should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of IoU calculation. Defaults to 'iou'.

        Returns:
            np.array: Calculated iou of boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), (
            '"boxes1" and "boxes2" should'
            f"be in the same type, got {type(boxes1)} and {type(boxes2)}."
        )

        boxes1_top_height = boxes1.top_height.reshape(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.reshape(-1, 1)
        boxes2_top_height = boxes2.top_height.reshape(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.reshape(1, -1)

        heighest_of_bottom = np.maximum(boxes1_bottom_height, boxes2_bottom_height)
        lowest_of_top = np.minimum(boxes1_top_height, boxes2_top_height)
        overlaps_h = np.maximum(lowest_of_top - heighest_of_bottom, 0)
        return overlaps_h

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode="iou"):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            np.array: Calculated 3D overlaps of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), (
            '"boxes1" and "boxes2" should'
            f"be in the same type, got {type(boxes1)} and {type(boxes2)}."
        )

        assert mode in ["iou", "iof"]

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return np.zeros((rows, cols))

        # height overlap
        overlaps_h = cls.height_overlaps(boxes1, boxes2)

        # bev overlap
        iou2d = box_iou_rotated(boxes1.bev, boxes2.bev)
        areas1 = (boxes1.bev[:, 2] * boxes1.bev[:, 3])[:, None]
        areas1 = np.broadcast_to(areas1, (rows, cols))
        areas2 = (boxes2.bev[:, 2] * boxes2.bev[:, 3])[None, :]
        areas2 = np.broadcast_to(areas2, (rows, cols))
        overlaps_bev = iou2d * (areas1 + areas2) / (1 + iou2d)

        # 3d overlaps
        overlaps_3d = overlaps_bev * overlaps_h

        volume1 = boxes1.volume.reshape(-1, 1)
        volume2 = boxes2.volume.reshape(1, -1)

        if mode == "iou":
            # the clamp func is used to avoid division of 0
            iou3d = overlaps_3d / np.maximum(volume1 + volume2 - overlaps_3d, 1e-8)
        else:
            iou3d = overlaps_3d / np.maximum(volume1, 1e-8)

        return iou3d

    def new_box(self, data):
        """Create a new box object with data.

        The new box and its tensor has the similar properties
            as self and self.tensor, respectively.

        Args:
            data (np.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``,
                the object's other properties are similar to ``self``.
        """
        new_tensor = np.array(data) if not isinstance(data, np.ndarray) else data
        original_type = type(self)
        return original_type(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def points_in_boxes_part(self, points, boxes_override=None):
        """Find the box in which each point is.

        Args:
            points (np.array): Points in shape (1, M, 3) or (M, 3),
                3 dimensions are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (np.array, optional): Boxes to override
                `self.tensor`. Defaults to None.

        Returns:
            np.array: The index of the first box that each point
                is in, in shape (M, ). Default value is -1
                (if the point is not enclosed by any box).

        Note:
            If a point is enclosed by multiple boxes, the index of the
            first box will be returned.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor
        if points.ndim == 2:
            points = points[None, ...]
        box_idx = points_in_boxes_part(points, boxes[None, ...]).squeeze(0)
        return box_idx

    def points_in_boxes_all(self, points, boxes_override=None):
        """Find all boxes in which each point is.

        Args:
            points (np.array): Points in shape (1, M, 3) or (M, 3),
                3 dimensions are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (np.array, optional): Boxes to override
                `self.tensor`. Defaults to None.

        Returns:
            np.array: A tensor indicating whether a point is in a box,
                in shape (M, T). T is the number of boxes. Denote this
                tensor as A, if the m^th point is in the t^th box, then
                `A[m, t] == 1`, elsewise `A[m, t] == 0`.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor

        points_clone = points.copy()[..., :3]
        if points_clone.ndim == 2:
            points_clone = points_clone[None, ...]
        else:
            assert points_clone.ndim == 3 and points_clone.shape[0] == 1

        boxes = boxes[None, ...]
        box_idxs_of_pts = points_in_boxes_all(points_clone, boxes)

        return box_idxs_of_pts.squeeze(0)

    def points_in_boxes(self, points, boxes_override=None):
        warnings.warn(
            "DeprecationWarning: points_in_boxes is a "
            "deprecated method, please consider using "
            "points_in_boxes_part."
        )
        return self.points_in_boxes_part(points, boxes_override)

    def points_in_boxes_batch(self, points, boxes_override=None):
        warnings.warn(
            "DeprecationWarning: points_in_boxes_batch is a "
            "deprecated method, please consider using "
            "points_in_boxes_all."
        )
        return self.points_in_boxes_all(points, boxes_override)


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                up z    x front (yaw=0)
                                   ^   ^
                                   |  /
                                   | /
       (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and increases from
    the positive direction of x to the positive direction of y.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (np.ndarray): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS = 2

    @property
    def gravity_center(self):
        """np.array: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = np.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """np.array: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
        if self.tensor.size == 0:
            return np.empty([0, 8, 3], dtype=self.tensor.dtype)

        dims = self.dims
        corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1).astype(
            dims.dtype
        )

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - np.array([0.5, 0.5, 0])
        corners = dims[:, None, :] * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3][:, None, :]
        return corners

    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angles (float | np.ndarray):
                Rotation angle or rotation matrix.
            points (np.ndarray | :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns
                None, otherwise it returns the rotated points and the
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, np.ndarray):
            angle = np.array(angle)

        assert (
            angle.shape == (3, 3) or angle.size == 1
        ), f"invalid rotation angle shape {angle.shape}"

        if angle.size == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3], angle, axis=self.YAW_AXIS, return_mat=True
            )
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, np.ndarray):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif hasattr(points, "rotate"):  # BasePoints-like object
                points.rotate(rot_mat_T)
            else:
                raise ValueError("Unsupported points type")
            return points, rot_mat_T

    def flip(self, bev_direction="horizontal", points=None):
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (np.ndarray | :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            np.ndarray or None: Flipped points.
        """
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        elif bev_direction == "vertical":
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi

        if points is not None:
            assert isinstance(points, np.ndarray) or hasattr(points, "flip")
            if isinstance(points, np.ndarray):
                if bev_direction == "horizontal":
                    points[:, 1] = -points[:, 1]
                elif bev_direction == "vertical":
                    points[:, 0] = -points[:, 0]
            elif hasattr(points, "flip"):  # BasePoints-like object
                points.flip(bev_direction)
            return points

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): the target Box mode
            rt_mat (np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`:
                The converted box of the same type in the ``dst`` mode.
        """
        # Note: This is a placeholder implementation
        # In a real implementation, you would need to import and use Box3DMode
        # from .box_3d_mode import Box3DMode
        # return Box3DMode.convert(box=self, src=Box3DMode.LIDAR, dst=dst, rt_mat=rt_mat)

        # For now, return a copy of self as a placeholder
        return self.clone()

    def enlarged_box(self, extra_width):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | np.ndarray): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.copy()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

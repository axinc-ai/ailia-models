"""General 3D Bounding Box class."""

import numpy as np
from numpy.linalg import lstsq as optimizer
from scipy.spatial.transform import Rotation as rotation_util

EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]  # lines along z-axis
)

# The vertices are ordered according to the left-hand rule, so the normal
# vector of each face will point inward the box.
FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])

UNIT_BOX = np.asarray([
    [0., 0., 0.],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
])

NUM_KEYPOINTS = 9
FRONT_FACE_ID = 4
TOP_FACE_ID = 2


class Box(object):
    """General 3D Oriented Bounding Box."""

    def __init__(self, vertices=None):
        if vertices is None:
            vertices = self.scaled_axis_aligned_vertices(np.array([1., 1., 1.]))

        self._vertices = vertices
        self._rotation = None
        self._translation = None
        self._scale = None
        self._transformation = None
        self._volume = None

    @classmethod
    def from_transformation(cls, rotation, translation, scale):
        """Constructs an oriented bounding box from transformation and scale."""
        if rotation.size != 3 and rotation.size != 9:
            raise ValueError('Unsupported rotation, only 3x1 euler angles or 3x3 ' +
                             'rotation matrices are supported. ' + rotation)
        if rotation.size == 3:
            rotation = rotation_util.from_rotvec(rotation.tolist()).as_dcm()
        scaled_identity_box = cls.scaled_axis_aligned_vertices(scale)
        vertices = np.zeros((NUM_KEYPOINTS, 3))
        for i in range(NUM_KEYPOINTS):
            vertices[i, :] = np.matmul(
                rotation, scaled_identity_box[i, :]) + translation.flatten()
        return cls(vertices=vertices)

    def __repr__(self):
        representation = 'Box: '
        for i in range(NUM_KEYPOINTS):
            representation += '[{0}: {1}, {2}, {3}]'.format(i, self.vertices[i, 0],
                                                            self.vertices[i, 1],
                                                            self.vertices[i, 2])
        return representation

    def __len__(self):
        return NUM_KEYPOINTS

    def __name__(self):
        return 'Box'

    def apply_transformation(self, transformation):
        """Applies transformation on the box.

        Group multiplication is the same as rotation concatenation. Therefore return
        new box with SE3(R * R2, T + R * T2); Where R2 and T2 are existing rotation
        and translation. Note we do not change the scale.

        Args:
          transformation: a 4x4 transformation matrix.

        Returns:
           transformed box.
        """
        if transformation.shape != (4, 4):
            raise ValueError('Transformation should be a 4x4 matrix.')

        new_rotation = np.matmul(transformation[:3, :3], self.rotation)
        new_translation = transformation[:3, 3] + (
            np.matmul(transformation[:3, :3], self.translation))
        return Box.from_transformation(new_rotation, new_translation, self.scale)

    @classmethod
    def scaled_axis_aligned_vertices(cls, scale):
        """Returns an axis-aligned set of verticies for a box of the given scale.

        Args:
          scale: A 3*1 vector, specifiying the size of the box in x-y-z dimension.
        """
        w = scale[0] / 2.
        h = scale[1] / 2.
        d = scale[2] / 2.

        # Define the local coordinate system, w.r.t. the center of the box
        aabb = np.array([[0., 0., 0.], [-w, -h, -d], [-w, -h, +d], [-w, +h, -d],
                         [-w, +h, +d], [+w, -h, -d], [+w, -h, +d], [+w, +h, -d],
                         [+w, +h, +d]])
        return aabb

    @classmethod
    def fit(cls, vertices):
        """Estimates a box 9-dof parameters from the given vertices.

        Directly computes the scale of the box, then solves for orientation and
        translation.

        Args:
          vertices: A 9*3 array of points. Points are arranged as 1 + 8 (center
            keypoint + 8 box vertices) matrix.

        Returns:
          orientation: 3*3 rotation matrix.
          translation: 3*1 translation vector.
          scale: 3*1 scale vector.
        """
        orientation = np.identity(3)
        translation = np.zeros((3, 1))
        scale = np.zeros(3)

        # The scale would remain invariant under rotation and translation.
        # We can safely estimate the scale from the oriented box.
        for axis in range(3):
            for edge_id in range(4):
                # The edges are stored in quadruples according to each axis
                begin, end = EDGES[axis * 4 + edge_id]
                scale[axis] += np.linalg.norm(vertices[begin, :] - vertices[end, :])
            scale[axis] /= 4.

        x = cls.scaled_axis_aligned_vertices(scale)
        system = np.concatenate((x, np.ones((NUM_KEYPOINTS, 1))), axis=1)
        solution, _, _, _ = optimizer(system, vertices, rcond=None)
        orientation = solution[:3, :3].T
        translation = solution[3, :3]
        return orientation, translation, scale

    def inside(self, point):
        """Tests whether a given point is inside the box.

          Brings the 3D point into the local coordinate of the box. In the local
          coordinate, the looks like an axis-aligned bounding box. Next checks if
          the box contains the point.
        Args:
          point: A 3*1 numpy vector.

        Returns:
          True if the point is inside the box, False otherwise.
        """
        inv_trans = np.linalg.inv(self.transformation)
        scale = self.scale
        point_w = np.matmul(inv_trans[:3, :3], point) + inv_trans[:3, 3]
        for i in range(3):
            if abs(point_w[i]) > scale[i] / 2.:
                return False
        return True

    def sample(self):
        """Samples a 3D point uniformly inside this box."""
        point = np.random.uniform(-0.5, 0.5, 3) * self.scale
        point = np.matmul(self.rotation, point) + self.translation
        return point

    @property
    def vertices(self):
        return self._vertices

    @property
    def rotation(self):
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._rotation

    @property
    def translation(self):
        if self._translation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._translation

    @property
    def scale(self):
        if self._scale is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._scale

    @property
    def volume(self):
        """Compute the volume of the parallelpiped or the box.

          For the boxes, this is equivalent to np.prod(self.scale). However for
          parallelpiped, this is more involved. Viewing the box as a linear function
          we can estimate the volume using a determinant. This is equivalent to
          sp.ConvexHull(self._vertices).volume

        Returns:
          volume (float)
        """
        if self._volume is None:
            i = self._vertices[2, :] - self._vertices[1, :]
            j = self._vertices[3, :] - self._vertices[1, :]
            k = self._vertices[5, :] - self._vertices[1, :]
            sys = np.array([i, j, k])
            self._volume = abs(np.linalg.det(sys))
        return self._volume

    @property
    def transformation(self):
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        if self._transformation is None:
            self._transformation = np.identity(4)
            self._transformation[:3, :3] = self._rotation
            self._transformation[:3, 3] = self._translation
        return self._transformation

    def get_ground_plane(self, gravity_axis=1):
        """Get ground plane under the box."""

        gravity = np.zeros(3)
        gravity[gravity_axis] = 1

        def get_face_normal(face, center):
            """Get a normal vector to the given face of the box."""
            v1 = self.vertices[face[0], :] - center
            v2 = self.vertices[face[1], :] - center
            normal = np.cross(v1, v2)
            return normal

        def get_face_center(face):
            """Get the center point of the face of the box."""
            center = np.zeros(3)
            for vertex in face:
                center += self.vertices[vertex, :]
            center /= len(face)
            return center

        ground_plane_id = 0
        ground_plane_error = 10.

        # The ground plane is defined as a plane aligned with gravity.
        # gravity is the (0, 1, 0) vector in the world coordinate system.
        for i in [0, 2, 4]:
            face = FACES[i, :]
            center = get_face_center(face)
            normal = get_face_normal(face, center)
            w = np.cross(gravity, normal)
            w_sq_norm = np.linalg.norm(w)
            if w_sq_norm < ground_plane_error:
                ground_plane_error = w_sq_norm
                ground_plane_id = i

        face = FACES[ground_plane_id, :]
        center = get_face_center(face)
        normal = get_face_normal(face, center)

        # For each face, we also have a parallel face that it's normal is also
        # aligned with gravity vector. We pick the face with lower height (y-value).
        # The parallel to face 0 is 1, face 2 is 3, and face 4 is 5.
        parallel_face_id = ground_plane_id + 1
        parallel_face = FACES[parallel_face_id]
        parallel_face_center = get_face_center(parallel_face)
        parallel_face_normal = get_face_normal(parallel_face, parallel_face_center)
        if parallel_face_center[gravity_axis] < center[gravity_axis]:
            center = parallel_face_center
            normal = parallel_face_normal
        return center, normal

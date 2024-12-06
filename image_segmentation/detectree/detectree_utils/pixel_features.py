"""Build pixel features."""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color, morphology, transform
from skimage.filters import rank

from . import filters

# to convert to illumination-invariant color space
# https://www.cs.harvard.edu/~sjg/papers/cspace.pdf
B = np.array(
    [
        [0.9465229, 0.2946927, -0.1313419],
        [-0.1179179, 0.9929960, 0.007371554],
        [0.09230461, -0.04645794, 0.9946464],
    ]
)

A = np.array(
    [
        [27.07439, -22.80783, -1.806681],
        [-5.646736, -7.722125, 12.86503],
        [-4.163133, -4.579428, -4.576049],
    ]
)

NUM_RGB_CHANNELS = 3
NUM_LAB_CHANNELS = 3
NUM_XYZ_CHANNELS = 3
NUM_ILL_CHANNELS = 3


class PixelFeaturesBuilder:
    """Customize how pixel features are computed."""

    def __init__(
        self,
        *,
        sigmas=[1, np.sqrt(2), 2],
        num_orientations=6,
        neighborhood=None,
        min_neighborhood_range=2,
        num_neighborhoods=3,
    ):
        """
        Initialize the pixel feature builder.

        See the `background <https://bit.ly/2KlCICO>`_ example notebook for more
        details.

        Parameters
        ----------
        sigmas : list-like, optional
            The list of scale parameters (sigmas) to build the Gaussian filter bank that
            will be used to compute the pixel-level features. The provided argument will
            be passed to the initialization method of the `PixelFeaturesBuilder`
            class. 
        num_orientations : int, optional
            The number of equally-distributed orientations to build the Gaussian filter
            bank that will be used to compute the pixel-level features. The provided
            argument will be passed to the initialization method of the
            `PixelFeaturesBuilder` class.
        neighborhood : array-like, optional
            The base neighborhood structure that will be used to compute the entropy
            features. The provided argument will be passed to the initialization method
            of the `PixelFeaturesBuilder` class. If no value is provided, a square with
            a side size of `2 * min_neighborhood_range + 1` is used.
        min_neighborhood_range : int, optional
            The range (i.e., the square radius) of the smallest neigbhorhood window that
            will be used to compute the entropy features. The provided argument will be
            passed to the initialization method of the `PixelFeaturesBuilder` class. 
        num_neighborhoods : int, optional
            The number of neigbhorhood windows (whose size follows a geometric
            progression starting at `min_neighborhood_range`) that will be used to
            compute the entropy features. The provided argument will be passed to the
            initialization method of the `PixelFeaturesBuilder` class.
        """
        if neighborhood is None:
            neighborhood = morphology.square(2 * min_neighborhood_range + 1)

        self.sigmas = sigmas
        self.num_orientations = num_orientations
        self.neighborhood = neighborhood
        self.scales = np.geomspace(
            1, 2 ** (num_neighborhoods - 1), num_neighborhoods
        ).astype(int)

        self.num_color_features = NUM_LAB_CHANNELS + NUM_ILL_CHANNELS
        self.num_texture_features = num_orientations * len(sigmas)
        self.num_entropy_features = num_neighborhoods
        self.num_pixel_features = (
            self.num_color_features
            + self.num_texture_features
            + self.num_entropy_features
        )

    def build_features_from_arr(self, img_rgb):
        """
        Build feature array from an RGB image array.

        Parameters
        ----------
        img_rgb : numpy ndarray
            The image in RGB format, i.e., in a 3-D array

        Returns
        -------
        responses : numpy ndarray
            Array with the pixel responses.
        """
        num_rows, num_cols, _ = img_rgb.shape
        num_pixels = num_rows * num_cols
        img_lab = color.rgb2lab(img_rgb)
        img_lab_l = img_lab[:, :, 0]  # ACHTUNG: this is a view

        X = np.zeros((num_pixels, self.num_pixel_features), dtype=np.float32)

        # color features
        img_lab_vec = img_lab.reshape(num_rows * num_cols, NUM_LAB_CHANNELS)
        img_xyz_vec = color.rgb2xyz(img_rgb).reshape(
            num_rows * num_cols, NUM_XYZ_CHANNELS
        )
        img_ill_vec = np.dot(
            A, np.log(np.dot(B, img_xyz_vec.transpose()) + 1)
        ).transpose()
        X[:, :NUM_LAB_CHANNELS] = img_lab_vec
        X[:, NUM_LAB_CHANNELS : NUM_LAB_CHANNELS + NUM_ILL_CHANNELS] = img_ill_vec

        # texture features
        for i, sigma in enumerate(self.sigmas):
            base_kernel_arr = filters.get_texture_kernel(sigma)
            for j, orientation in enumerate(range(self.num_orientations)):
                theta = orientation * 180 / self.num_orientations
                oriented_kernel_arr = ndi.rotate(base_kernel_arr, theta)
                img_filtered = cv2.filter2D(
                    img_lab_l, ddepth=-1, kernel=oriented_kernel_arr
                )
                img_filtered_vec = img_filtered.flatten()
                X[:, self.num_color_features + i * self.num_orientations + j] = (
                    img_filtered_vec
                )

        # entropy features
        entropy_start = self.num_color_features + self.num_texture_features
        X[:, entropy_start] = rank.entropy(
            img_lab_l.astype(np.uint16), self.neighborhood
        ).flatten()

        for i, factor in enumerate(self.scales[1:], start=1):
            img = transform.resize(
                transform.downscale_local_mean(img_lab_l, (factor, factor)),
                img_lab_l.shape,
            ).astype(np.uint16)
            X[:, entropy_start + i] = rank.entropy(img, self.neighborhood).flatten()

        return X

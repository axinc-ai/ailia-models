try:
    import accimage
except ImportError:
    accimage = None

import collections
import numbers
import numpy as np
from PIL import Image


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.NEAREST):
        assert (
            isinstance(size, int)
            or isinstance(size, float)
            or (isinstance(size, collections.Iterable) and len(size) == 2)
        )
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        if img.ndim == 3:
            return np.array(Image.fromarray(img).resize(self.size, self.interpolation))
        elif img.ndim == 2:
            return np.array(Image.fromarray(img).resize(self.size, self.interpolation))
        else:
            RuntimeError(
                f"img should be ndarray with 2 or 3 dimensions. Got {img.ndim}."
            )


class CenterCrop(object):
    """Crops the given ``numpy.ndarray`` at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for center crop.

        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))

        # # randomized cropping
        # i = np.random.randint(i-3, i+4)
        # j = np.random.randint(j-3, j+4)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.

        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)

        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError(f"img should be ndarray. Got {type(img)}.")
        if img.ndim == 3:
            return img[i : i + h, j : j + w, :]
        elif img.ndim == 2:
            return img[i : i + h, j : j + w]
        else:
            raise RuntimeError(
                f"img should be ndarray with 2 or 3 dimensions. Got {img.ndim}."
            )

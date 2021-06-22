try:
    import accimage
except ImportError:
    accimage = None

import collections
import numbers
import numpy as np
from PIL import Image, ImageEnhance
import scipy.ndimage.interpolation as itpl
import types


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}.")

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}.")

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}.")

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor {hue_factor} is not in [-0.5, 0.5].")

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}.")

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img

    h, s, v = img.convert("HSV").split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")

    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}.")

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")

    input_mode = img.mode
    img = img.convert("RGB")

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255) ** gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, "RGB").convert(input_mode)
    return img


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


class NormalizeNumpyArray(object):
    """Normalize a ``numpy.ndarray`` with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(M1,..,Mn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``numpy.ndarray`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image of size (H, W, C) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError(f"img should be ndarray. Got {type(img)}.")
        # TODO: make efficient
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img


class Rotate(object):
    """Rotates the given ``numpy.ndarray``.

    Args:
        angle (float): The rotation angle in degrees.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be rotated.

        Returns:
            img (numpy.ndarray (C x H x W)): Rotated image.
        """

        # order=0 means nearest-neighbor type interpolation
        return itpl.rotate(img, self.angle, reshape=False, prefilter=False, order=0)


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


class BottomCrop(object):
    """Crops the given ``numpy.ndarray`` at the bottom.

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
        """Get parameters for ``crop`` for bottom crop.

        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for bottom crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = h - th
        j = int(round((w - tw) / 2.0))

        # randomized left and right cropping
        # i = np.random.randint(i-3, i+4)
        # j = np.random.randint(j-1, j+1)

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


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class HorizontalFlip(object):
    """Horizontally flip the given ``numpy.ndarray``.

    Args:
        do_flip (boolean): whether or not do horizontal flip.

    """

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be flipped.

        Returns:
            img (numpy.ndarray (C x H x W)): flipped image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError(f"img should be ndarray. Got {type(img)}.")

        if self.do_flip:
            return np.fliplr(img)
        else:
            return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(
                max(0, 1 - brightness), 1 + brightness
            )
            transforms.append(
                Lambda(lambda img: adjust_brightness(img, brightness_factor))
            )

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(
                max(0, 1 - saturation), 1 + saturation
            )
            transforms.append(
                Lambda(lambda img: adjust_saturation(img, saturation_factor))
            )

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Input image.

        Returns:
            img (numpy.ndarray (C x H x W)): Color jittered image.
        """
        if not (_is_numpy_image(img)):
            raise TypeError(f"img should be ndarray. Got {type(img)}.")

        pil = Image.fromarray(img)
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        return np.array(transform(pil))


class Crop(object):
    """Crops the given PIL Image to a rectangular region based on a given
    4-tuple defining the left, upper pixel coordinated, hight and width size.

    Args:
        a tuple: (upper pixel coordinate, left pixel coordinate, hight, width)-tuple
    """

    def __init__(self, i, j, h, w):
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """

        i, j, h, w = self.i, self.j, self.h, self.w

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

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(i={self.i},j={self.j},h={self.h},w={self.w})"
        )

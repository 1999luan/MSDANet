import numpy as np
import random
from PIL import Image
import scipy.ndimage
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numbers

def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class RandomZoom(object):
    def __init__(self, zoom=(0.8, 1.2)):
        self.min, self.max = zoom[0], zoom[1]

    def __call__(self, image, target):

        if random.random() < 0.5:
            zoom = random.uniform(self.min, self.max)
            image_mode = image.mode
            image = np.array(image)
            zoom_v = clipped_zoom(image, zoom)
            zoom_v = Image.fromarray(zoom_v.astype('uint8'), image_mode)
            image = zoom_v
            target_mode = target.mode
            target = np.array(target)
            zoom_k = clipped_zoom(target, zoom)
            zoom_k = Image.fromarray(zoom_k.astype('uint8'), target_mode)
            target = zoom_k

        return image, target

class RandomRotation(object):

    def __init__(self, degrees=90, resample=None, expand=False, center=None):
        if isinstance(degrees,numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, image, target):

        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        if random.random() < 0.5:
            angle = self.get_params(self.degrees)
            image = F.rotate(image, angle=angle, resample=self.resample, expand=self.expand, center=self.center)
            target = F.rotate(target, angle=angle, resample=self.resample, expand=self.expand, center=self.center)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        zoom_in = scipy.ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `zoom_in` might still be slightly different with `img` due to rounding, so
        # trim off any extra pixels at the edges or zero-padding

        if zoom_in.shape[0] >= h:
            zoom_top = (zoom_in.shape[0] - h) // 2
            sh = h
            out_top = 0
            oh = h
        else:
            zoom_top = 0
            sh = zoom_in.shape[0]
            out_top = (h - zoom_in.shape[0]) // 2
            oh = zoom_in.shape[0]
        if zoom_in.shape[1] >= w:
            zoom_left = (zoom_in.shape[1] - w) // 2
            sw = w
            out_left = 0
            ow = w
        else:
            zoom_left = 0
            sw = zoom_in.shape[1]
            out_left = (w - zoom_in.shape[1]) // 2
            ow = zoom_in.shape[1]

        out = np.zeros_like(img)
        out[out_top:out_top + oh, out_left:out_left + ow] = zoom_in[zoom_top:zoom_top + sh, zoom_left:zoom_left + sw]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out
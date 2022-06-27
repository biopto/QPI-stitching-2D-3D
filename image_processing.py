__author__ = 'Piotr Stępień'

import h5py
import numpy as np
from skimage.transform import resize
from skimage.util import img_as_float


def magnitude_of_gradient(im):
    grad_y, grad_x = np.gradient(im)
    return np.sqrt(np.square(grad_x) + np.square(grad_y))


def tv(im):
    grad_y, grad_x = np.gradient(im)
    return np.sqrt(np.square(grad_x) + np.square(grad_y)).sum()


def rescale_to_phase(im, prev_min, prev_max, bits=16):
    max_val = 2**bits - 1
    curr_min = np.nanmin(im)
    # im = im - curr_min
    im = im / max_val * (prev_max - prev_min)
    im = im + prev_min
    return im


def convert_int16_to_n(vol):
    # vol_new = np.asarray(vol.copy(), dtype=np.float16)
    vol_new = vol
    # vol_new = np.asarray(vol_new, dtype=np.float)
    vol_new = vol_new / (2**16 - 1)
    vol_new = vol_new + 1
    return vol_new


def convert_fused_hdf5_to_array(file_path, prev_min, prev_max, bits=16, resolution=0, zero_to_nan=True):
    file = h5py.File(str(file_path), 'r')
    # downsampling_factor = file['/s00/resolutions'][()][resolution] # 1 means no downsampling
    ph = file[f'/t00000/s00/{resolution}/cells'][()]
    ph = ph[0, :, :]
    ph = ph.astype(np.float)
    max_value = 0
    if bits == 16:
        max_value = np.iinfo(np.uint16).max
    # elif bits == 32:
    #     max_value = np.finfo(np.float32).max
    ph[ph < 0] += max_value
    if zero_to_nan is True:
        ph[ph == 0] = np.nan
    ph = rescale_to_phase(ph, prev_min, prev_max, bits)
    return ph


def set_range(im, lower_b, upper_b):
    range_val = upper_b - lower_b
    im_max = np.max(im)
    im_min = np.min(im)
    if im_max == 0:
        im_max += 1
        im_min += 1
        im += 1
    im = im - im_min
    im = im / (im_max - im_min)
    im = im * range_val + lower_b
    return im, im_min, im_max


def set_reset_range(im, lower_b, upper_b, fun, *args, **kwargs):
    im_mod, prev_min, prev_max = set_range(im, lower_b, upper_b)
    im_mod = fun(im_mod, *args, **kwargs)
    im_mod, _, _ = set_range(im_mod, prev_min, prev_max)
    return im_mod


def resize_mat(im, new_shape):
    im_mod, prev_min, prev_max = set_range(im, 0, 1)
    im_mod = resize(im_mod, new_shape)
    im_mod, _, _ = set_range(im_mod, prev_min, prev_max)
    return im_mod


def ims_range(ims):
    ims_max = np.max(ims)
    ims_min = np.min(ims)
    return ims_max - ims_min


def batch_round(nums, round_to_digit=None):
    rounded = []
    for num in nums:
        rounded.append(round(num, round_to_digit))
    return rounded


def im_preview(im, lower=0, upper=1):
    im_2, _, _ = set_range(im, lower, upper)
    # im_2 = im_2.astype(np.float32)
    return np.asarray([im_2, im]).astype('float32')


def rmse(im1, im2):
    return np.sqrt(np.mean((im1 - im2) ** 2))


def mse(im1, im2):
    return np.mean((im1 - im2) ** 2)

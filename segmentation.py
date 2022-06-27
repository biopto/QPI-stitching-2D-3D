__author__ = 'Piotr Stępień'

from matplotlib.pyplot import sci
import scipy
from skimage.exposure import rescale_intensity
from skimage.filters.rank import gradient
from skimage.filters import threshold_li
from skimage.morphology import area_closing, binary_opening
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage
from image_processing import magnitude_of_gradient


def segment_cells_ws(im=None, background_g=1, cell_g=5, orig_min_max=(-8.486898, 14.647226)):
    im_rescaled = rescale_intensity(im, in_range=orig_min_max, out_range=(0, 1))
    im_g = gradient(im_rescaled, selem=np.ones((3, 3)))
    im_markers = np.zeros_like(im_g)
    im_markers[im_g < background_g] = 1  # background
    im_markers[im_g > cell_g] = 2  # cells
    segments = watershed(im_g, im_markers)
    segments = np.asarray(segments - 1, dtype=np.bool)
    percentage = im.shape[0] * im.shape[1] // 100
    segments = area_closing(segments, area_threshold=percentage)
    segments = binary_opening(segments, np.ones((5, 5)))
    return segments


def unwrapping_error_segmentation(img):
    im = img.copy()
    im_mog = magnitude_of_gradient(im)
    im_mog[im_mog < 1.2] = 0
    try:
        val_mog = threshold_li(im_mog)
        im_mog_b = np.zeros_like(im)
        im_mog_b[im_mog > val_mog] = 1
        # im_mog_b = ndimage.binary_dilation(im_mog_b, structure=np.ones((15, 15)))
        im_mog_b = ndimage.binary_dilation(im_mog_b, structure=np.ones((17, 17)))
        im_mog_b = ndimage.binary_fill_holes(im_mog_b)
    except ValueError:
        im_mog_b = np.zeros_like(im)
    except UnboundLocalError:
        test = 1
    im[im_mog_b == True] = np.median(im)
    return im_mog_b, im

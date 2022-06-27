__author__ = 'Piotr Stępień'

import numpy as np
import matplotlib.pyplot as plt

# from numba import jit
import imagej
import copy
import os
from numpy.lib.function_base import diff
from scipy.io.matlab.mio import savemat
from skimage.io import imread, imsave
from pathlib import Path
from skimage.feature import canny  # , register_translation

import skimage.registration._phase_cross_correlation as register_translation
from skimage.io.collection import imread_collection_wrapper
from skimage.util import dtype
from aberrations import (
    gradient_PF,
    legendre_fitting_nan,
    svd_correction,
    plane_fitting,
    slope_fitting,
    magnitude_of_gradient,
)
from scipy import ndimage, stats
from skimage import filters
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.exposure import rescale_intensity
from skimage.filters.rank import gradient
from skimage.morphology import area_closing, binary_opening
from skimage.segmentation import watershed
from skimage.filters.thresholding import threshold_otsu
from scipy.optimize import minimize
from image_processing import set_range, rmse, mse, tv
from utils_3d import sub_mat_fov
from visualization import subplot_row
from segmentation import unwrapping_error_segmentation, segment_cells_ws


class ImageCollection:
    def __init__(self, paths, params):
        # self.images = [[Image(paths[row][col], row, col) for col in range(cols)] for row in range(rows)]
        self.params = params
        self.paths = paths
        if 'sign' in params:
            self.sign = params['sign']
        else:
            self.sign = 1
        if 'overlap' in params:
            self.overlap = params['overlap']
        else:
            self.overlap = 0
        if 'rows' in params:
            self.rows = params['rows']
        else:
            self.rows = 1
        if 'cols' in params:
            self.cols = params['cols']
        else:
            self.cols = 1
        self.mean_aberr = np.zeros((self.rows, self.cols))
        self.offsets = np.zeros(self.cols * self.rows)
        self.planes_params = np.zeros(self.cols * self.rows * 3)
        if 'px_size' in params:
            self.px_size = params['px_size']
        else:
            self.px_size = {'x': 1, 'y': 1}
        if 'description' in params:
            self.description = params['description']
        else:
            self.description = ''
        self.abr_path = Path(os.path.basename(self.paths[0][0])[:16] + '.tiff')
        self.abr_path = Path(os.path.dirname(self.paths[0][0]) / self.abr_path)

    def create_copy(self, idx=2):
        self_copy = copy.deepcopy(self)
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(self.paths[row][col])
                path_new = self_copy.paths[row][col]
                path_new = path_new.parents[0] / (path_new.stem + f'_{idx}' + path_new.suffix)
                self_copy.paths[row][col] = path_new
                imsave(path_new, im)
        return self_copy

    def image(self, row, col, save_path=None):
        im = imread(self.paths[row][col])
        if save_path is not None:
            savemat(save_path, {'im': im})
        return im

    def create_tile_config(self, time_point=0, name='tileConfig.txt'):
        im = imread(self.paths[0][0])
        height, width = im.shape
        with open(str(self.paths[0][0].parent / name), 'w') as f:
            f.write('dim=3')
            f.write('\n')
            for col in range(self.cols):
                for row in range(self.rows):
                    pos_x = col * width * (1 - self.overlap) * self.px_size['x']
                    pos_y = row * height * (1 - self.overlap) * self.px_size['y']
                    idx = row + col * self.rows
                    f.write('{};{};({:.2f}, {:.2f}, 0)'.format(idx, time_point, pos_x, pos_y))
                    f.write('\n')

    def create_tile_config_yx(self, time_point=0, name='tileConfig.txt', reverse_x=False, reverse_y=False):
        im = imread(self.paths[0][0])
        height, width = im.shape
        if reverse_x is True:
            sign_x = -1
        else:
            sign_x = 1
        if reverse_y is True:
            sign_y = -1
        else:
            sign_y = 1
        with open(str(self.paths[0][0].parent / name), 'w') as f:
            f.write('dim=3')
            f.write('\n')
            for row in range(self.rows):
                for col in range(self.cols):
                    pos_x = col * width * (1 - self.overlap) * self.px_size['x'] * sign_x
                    pos_y = row * height * (1 - self.overlap) * self.px_size['y'] * sign_y
                    idx = row * self.cols + col
                    f.write('{};{};({:.2f}, {:.2f}, 0)'.format(idx, time_point, pos_x, pos_y))
                    f.write('\n')

    def concatenate_row(self, row):
        temp_row = None
        for col in range(self.cols):
            im = imread(self.paths[row][col])
            if temp_row is not None:
                temp_row = np.concatenate((temp_row, im), axis=1)
            else:
                temp_row = im.copy()
        return temp_row

    def arrange_grid(self, start_row=0, stop_row=None, start_col=0, stop_col=None, crop=False, corrections=True):
        # Arrange frames grid by loading frames from disk
        if stop_row is None:
            stop_row = self.rows
        else:
            stop_row = np.mod(stop_row, self.rows)
        if stop_col is None:
            stop_col = self.cols
        else:
            stop_col = np.mod(stop_col, self.cols)
        im_full = None
        planes_params = self.planes_params
        for row in range(self.rows):
            if start_row <= row <= stop_row:
                temp_row = None
                for col in range(self.cols):
                    if start_col <= col <= stop_col:
                        im = imread(self.paths[row][col])
                        if corrections:
                            if planes_params.any():
                                idx3 = (col + row * self.cols) * 3
                                coeffs = (planes_params[idx3], planes_params[idx3 + 1], planes_params[idx3 + 2])
                                plane = self._calc_plane(coeffs, override_downsample=True)
                                im += plane
                            else:
                                idx = col + row * self.cols
                                im += self.offsets[idx]
                        if crop:
                            margin_y = round(self.height * self.overlap)
                            margin_x = round(self.width * self.overlap)
                            im = im[:-margin_y, :-margin_x]
                        if temp_row is not None:
                            temp_row = np.concatenate((temp_row, im), axis=1)
                        else:
                            temp_row = im.copy()
                if im_full is not None:
                    im_full = np.concatenate((im_full, temp_row), axis=0)
                else:
                    im_full = temp_row.copy()
        return im_full

    def arrange_grid_memory(self, planes_params=None, drop_overlap=False):
        # Arrange grid of frames from memory
        im_full = None
        for row in range(self.rows):
            temp_row = None
            for col in range(self.cols):
                im = self.frames[row][col].copy()
                if planes_params is not None and np.any(planes_params):
                    idx3 = (col + row * self.cols) * 3
                    coeffs = (planes_params[idx3], planes_params[idx3 + 1], planes_params[idx3 + 2])
                    plane = self._calc_plane(coeffs, override_downsample=True)
                    im = im + plane
                if drop_overlap:
                    if type(drop_overlap) == bool:
                        overlap = self.overlap
                    elif type(drop_overlap) == float:
                        assert (drop_overlap > 0) and (drop_overlap < 1)
                        overlap = drop_overlap
                    margin_lr = round(overlap * self.width / 2)
                    margin_td = round(overlap * self.height / 2)
                    im = im[margin_td:-margin_td, margin_lr:-margin_lr]
                if temp_row is not None:
                    temp_row = np.concatenate((temp_row, im), axis=1)
                else:
                    temp_row = im.copy()
            if im_full is not None:
                im_full = np.concatenate((im_full, temp_row), axis=0)
            else:
                im_full = temp_row.copy()
        return im_full

    def reverse_x(self):
        for col in self.paths:
            col.reverse()
        if hasattr(self, 'frames'):
            for f in self.frames:
                f.reverse()
            for p in self.frames_paths:
                p.reverse()

    def reverse_y(self):
        self.paths.reverse()
        if hasattr(self, 'frames'):
            self.frames.reverse()
            self.frames_paths.reverse()

    def find_min_max(self):
        min_glob = 10000
        max_glob = -10000
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                min_curr = im.min()
                max_curr = im.max()
                if min_curr < min_glob:
                    min_glob = min_curr
                if max_curr > max_glob:
                    max_glob = max_curr
        return min_glob, max_glob

    def rescale_to_unsigned(self, min_o=-10, max_o=10, bits=16):
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im[im > max_o] = max_o
                im[im < min_o] = min_o
                max_val = 2**bits - 1
                im = im - min_o
                im = im / (max_o - min_o) * max_val
                if bits == 8:
                    im = im.astype(np.uint8)
                elif bits == 16:
                    im = im.astype(np.uint16)
                elif bits == 32:
                    im = im.astype(np.uint32)
                elif bits == 64:
                    im = im.astype(np.uint64)
                else:
                    os.error('Only 8, 6, 32 or 64 bits are allowed.')
                imsave(str(self.paths[row][col]), im)

    def parse_coords(self):
        self.xpos = np.zeros((self.rows, self.cols))
        self.ypos = np.zeros_like(self.xpos)
        coords_path = os.path.dirname(self.paths[0][0])
        coords_path = os.path.join(coords_path, 'TileConfiguration.registered.txt')
        file = open(coords_path)
        for line in file:
            if 'phase' in line:
                col = int(line[7:9].lstrip('0'))
                row = int(line[12:14].lstrip('0'))
                lpar = line.find('(')
                comma = line.find(',')
                rpar = line.find(')')
                x = float(line[lpar + 1 : comma])
                y = float(line[comma + 2 : rpar])
                self.xpos[row, col] = x
                self.ypos[row, col] = y

    def cut_side(self, img, side):
        if side == 'top' or side == 'bottom':
            im_size = round(img.shape[0] * self.overlap)
            if side == 'top':
                res = img[:im_size, :]
            else:
                res = img[-im_size:, :]
        elif side == 'left' or side == 'right':
            im_size = round(img.shape[1] * self.overlap)
            if side == 'left':
                res = img[:, :im_size]
            else:
                res = img[:, -im_size:]
        return res

    def cut_side_n(self, img, side):
        if side == 'top' or side == 'bottom':
            im_size = round(img.shape[0] * self.overlap)
            if side == 'top':
                res = img[im_size : im_size * 2, :]
            else:
                res = img[-2 * im_size : -im_size, :]
        elif side == 'left' or side == 'right':
            im_size = round(img.shape[1] * self.overlap)
            if side == 'left':
                res = img[:, im_size : im_size * 2]
            else:
                res = img[:, -2 * im_size : -im_size]
        return res

    def normalize_offset(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if col > 0:
                    im_0 = imread(str(self.paths[row][col]))
                    im_left = imread(str(self.paths[row][col - 1]))
                    overlap_diff = self.cut_side(im_0, 'left') - self.cut_side(im_left, 'right')
                elif row != 0:
                    im_0 = imread(str(self.paths[row][col]))
                    im_top = imread(str(self.paths[row - 1][col]))
                    overlap_diff = self.cut_side(im_0, 'top') - self.cut_side(im_top, 'bottom')
                else:
                    im_0 = imread(str(self.paths[row][col]))
                    overlap_diff = 0
                offset = np.mean(overlap_diff)
                # offset = np.median(overlap_diff)
                im_0 -= offset
                imsave(str(self.paths[row][col]), im_0)
        self.description += '_offset'
        print('Saved after offset correction')

    def normalize_offset_reverse(self):
        for col in range(self.cols):
            for row in range(self.rows):
                if col < self.cols:
                    im_0 = imread(str(self.paths[row][col]))
                    im_right = imread(str(self.paths[row][col + 1]))
                    overlap_diff = self.cut_side(im_0, 'right') - self.cut_side(im_right, 'left')
                elif row < self.rows + 1:
                    im_0 = imread(str(self.paths[row][col]))
                    im_bottom = imread(str(self.paths[row + 1][col]))
                    overlap_diff = self.cut_side(im_0, 'bottom') - self.cut_side(im_bottom, 'top')
                else:
                    im_0 = imread(str(self.paths[row][col]))
                    overlap_diff = 0
                offset = np.mean(overlap_diff)
                # offset = np.median(overlap_diff)
                im_0 -= offset
                imsave(str(self.paths[row][col]), im_0)
        self.description += '_offset'
        print('Saved after offset correction')

    def normalize_offset_2(self, fun=np.median):
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im_0 = np.copy(im)
                im_0_top = self.cut_side(im_0, 'top')
                im_0_left = self.cut_side(im_0, 'left')
                im_0_tl_corner = self.cut_side(im_0_left, 'top')
                if col > 0 and row > 0:
                    im_n_top = imread(str(self.paths[row - 1][col]))
                    im_n_left = imread(str(self.paths[row][col - 1]))
                    im_n_tl_corner = imread(str(self.paths[row - 1][col - 1]))
                    im_n_top = self.cut_side(im_n_top, 'bottom')
                    im_n_left = self.cut_side(im_n_left, 'right')
                    im_n_tl_corner = self.cut_side(self.cut_side(im_n_tl_corner, 'right'), 'bottom')
                    im_0_top_cut, im_n_top_cut = correlate_images(im_0_top, im_n_top)
                    im_0_left_cut, im_n_left_cut = correlate_images(im_0_left, im_n_left)
                    im_0_tl_corner_cut, im_n_tl_corner_cut = correlate_images(im_0_tl_corner, im_n_tl_corner)
                    diff_top = np.reshape(im_0_top_cut - im_n_top_cut, -1)
                    diff_left = np.reshape(im_0_left_cut - im_n_left_cut, -1)
                    diff_rl_corner = np.reshape(im_0_tl_corner_cut - im_n_tl_corner_cut, -1)
                    diff = np.concatenate((diff_top, diff_left, diff_rl_corner))
                elif col > 0 and row == 0:
                    im_n_left = imread(str(self.paths[row][col - 1]))
                    im_n_left = self.cut_side(im_n_left, 'right')
                    im_0_left_cut, im_n_left_cut = correlate_images(im_0_left, im_n_left)
                    diff = np.reshape(im_0_left_cut - im_n_left_cut, -1)
                elif col == 0 and row > 0:
                    im_n_top = imread(str(self.paths[row - 1][col]))
                    im_n_top = self.cut_side(im_n_top, 'bottom')
                    im_0_top_cut, im_n_top_cut = correlate_images(im_0_top, im_n_top)
                    diff = np.reshape(im_0_top_cut - im_n_top_cut, -1)
                else:
                    diff = np.zeros((2,))
                diff = diff[np.isfinite(diff)]
                offset = fun(diff)
                im -= offset
                imsave(str(self.paths[row][col]), im)
        # self.description += '_offset2'
        print('Saved after offset correction')

    def normalize_offset_3(self, thr_fun=np.nanpercentile):
        im_thr = None
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im_0 = np.copy(im)
                im_0_top = self.cut_side(im_0, 'top')
                im_0_left = self.cut_side(im_0, 'left')
                im_0_tl_corner = self.cut_side(im_0_left, 'top')
                if col > 0 and row > 0:
                    im_n_top = imread(str(self.paths[row - 1][col]))
                    im_n_left = imread(str(self.paths[row][col - 1]))
                    im_n_tl_corner = imread(str(self.paths[row - 1][col - 1]))
                    im_n_top = self.cut_side(im_n_top, 'bottom')
                    im_n_left = self.cut_side(im_n_left, 'right')
                    im_n_tl_corner = self.cut_side(self.cut_side(im_n_tl_corner, 'right'), 'bottom')
                    im_0_top_cut, im_n_top_cut = correlate_images(im_0_top, im_n_top)
                    im_0_left_cut, im_n_left_cut = correlate_images(im_0_left, im_n_left)
                    im_0_tl_corner_cut, im_n_tl_corner_cut = correlate_images(im_0_tl_corner, im_n_tl_corner)
                    im_0_top_val, im_n_top_val = threshold_2(
                        [im_0_top_cut, im_n_top_cut], thr_fun=thr_fun, im_thr=im_thr
                    )
                    im_0_left_val, im_n_left_val = threshold_2(
                        [im_0_left_cut, im_n_left_cut], thr_fun=thr_fun, im_thr=im_thr
                    )
                    im_0_tl_corner_val, im_n_tl_corner_val = threshold_2(
                        [im_0_tl_corner_cut, im_n_tl_corner_cut], thr_fun=thr_fun, im_thr=im_thr
                    )
                    diff_top = im_0_top_val - im_n_top_val
                    diff_left = im_0_left_val - im_n_left_val
                    diff_rl_corner = im_0_tl_corner_val - im_n_tl_corner_val
                    diff = np.asarray([diff_top, diff_left, diff_rl_corner])
                    diff = np.average(diff, weights=[1, 1, self.overlap])
                elif col > 0 and row == 0:
                    im_n_left = imread(str(self.paths[row][col - 1]))
                    im_n_left = self.cut_side(im_n_left, 'right')
                    im_0_left_cut, im_n_left_cut = correlate_images(im_0_left, im_n_left)
                    im_0_left_val, im_n_left_val = threshold_2(
                        [im_0_left_cut, im_n_left_cut], thr_fun=thr_fun, im_thr=im_thr
                    )
                    diff = im_0_left_val - im_n_left_val
                elif col == 0 and row > 0:
                    im_n_top = imread(str(self.paths[row - 1][col]))
                    im_n_top = self.cut_side(im_n_top, 'bottom')
                    im_0_top_cut, im_n_top_cut = correlate_images(im_0_top, im_n_top)
                    im_0_top_val, im_n_top_val = threshold_2(
                        [im_0_top_cut, im_n_top_cut], thr_fun=thr_fun, im_thr=im_thr
                    )
                    diff = im_0_top_val - im_n_top_val
                else:
                    diff = 0
                im -= diff
                imsave(str(self.paths[row][col]), im)
        # self.description += '_offset3'
        print('Saved after offset correction')

    def normalize_offset_4(self, fun=np.median):
        # Simple comparison without correlation
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im_0 = np.copy(im)
                im_0_top = self.cut_side(im_0, 'top')
                im_0_left = self.cut_side(im_0, 'left')
                im_0_tl_corner = self.cut_side(im_0_left, 'top')
                if col > 0 and row > 0:
                    im_n_top = imread(str(self.paths[row - 1][col]))
                    im_n_left = imread(str(self.paths[row][col - 1]))
                    im_n_tl_corner = imread(str(self.paths[row - 1][col - 1]))
                    im_n_top = self.cut_side(im_n_top, 'bottom')
                    im_n_left = self.cut_side(im_n_left, 'right')
                    im_n_tl_corner = self.cut_side(self.cut_side(im_n_tl_corner, 'right'), 'bottom')
                    diff_top = np.reshape(im_0_top - im_n_top, -1)
                    diff_left = np.reshape(im_0_left - im_n_left, -1)
                    diff_rl_corner = np.reshape(im_0_tl_corner - im_n_tl_corner, -1)
                    diff = np.concatenate((diff_top, diff_left, diff_rl_corner))
                elif col > 0 and row == 0:
                    im_n_left = imread(str(self.paths[row][col - 1]))
                    im_n_left = self.cut_side(im_n_left, 'right')
                    diff = np.reshape(im_0_left - im_n_left, -1)
                elif col == 0 and row > 0:
                    im_n_top = imread(str(self.paths[row - 1][col]))
                    im_n_top = self.cut_side(im_n_top, 'bottom')
                    diff = np.reshape(im_0_top - im_n_top, -1)
                else:
                    diff = np.zeros((2,))
                diff = diff[np.isfinite(diff)]
                offset = fun(diff)
                im -= offset
                imsave(str(self.paths[row][col]), im)
        # self.description += '_offset2'
        print('Saved after offset correction')

    def normalize_offset_5(self):
        # normalize_offset first followed by normalized offset per row
        self.normalize_offset()
        for row in range(self.rows - 1):
            row_0 = self.concatenate_row(row)
            row_1 = self.concatenate_row(row + 1)
            row_0_bottom = self.cut_side(row_0, 'bottom')
            row_1_top = self.cut_side(row_1, 'top')
            diff_row = np.reshape(row_0_bottom - row_1_top, -1)
            diff_row = diff_row[np.isfinite(diff_row)]
            offset = np.mean(diff_row)
            for col in range(self.cols):
                im = imread(str(self.paths[row + 1][col]))
                im += offset
                imsave(str(self.paths[row + 1][col]), im)
        print('Saved after offset correction')

    def load_frames_sides_from_frames(self):
        self.frames_left = [
            [self.cut_side(self.frames[row][col], 'left') for col in range(self.cols)] for row in range(self.rows)
        ]
        self.frames_right = [
            [self.cut_side(self.frames[row][col], 'right') for col in range(self.cols)] for row in range(self.rows)
        ]
        self.frames_top = [
            [self.cut_side(self.frames[row][col], 'top') for col in range(self.cols)] for row in range(self.rows)
        ]
        self.frames_bottom = [
            [self.cut_side(self.frames[row][col], 'bottom') for col in range(self.cols)] for row in range(self.rows)
        ]
        self.planes = [[np.zeros_like(self.frames[row][col]) for col in range(self.cols)] for row in range(self.rows)]
        self.planes_left = [
            [np.zeros_like(self.frames_left[row][col]) for col in range(self.cols)] for row in range(self.rows)
        ]
        self.planes_right = [
            [np.zeros_like(self.frames_right[row][col]) for col in range(self.cols)] for row in range(self.rows)
        ]
        self.planes_top = [
            [np.zeros_like(self.frames_top[row][col]) for col in range(self.cols)] for row in range(self.rows)
        ]
        self.planes_bottom = [
            [np.zeros_like(self.frames_bottom[row][col]) for col in range(self.cols)] for row in range(self.rows)
        ]

    def load_frames(self, downsample_rate=1, suffix=None):
        if hasattr(self, 'frames_paths'):
            frames_paths = self.frames_paths
        else:
            frames_paths = self.paths
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                print(f'load_frames: {i} / {self.rows * self.cols}')
                frame_path = frames_paths[row][col]
                if suffix is not None:
                    frame_path = frame_path.parent / (frame_path.stem + '_' + suffix + frame_path.suffix)
                if downsample_rate > 1:
                    factors = (downsample_rate, downsample_rate)
                    self.frames[row][col] = downscale_local_mean(imread(frame_path), factors)
                else:
                    self.frames[row][col] = imread(frame_path)
                self.frames[row][col] = self.frames[row][col].astype(np.float64)
        self.load_frames_sides_from_frames()

    def load_mogs(self):
        mogs = [[magnitude_of_gradient(self.frames[row][col]) for col in range(self.cols)] for row in range(self.rows)]
        self.mogs_left = [
            [self.cut_side(mogs[row][col], 'left') for col in range(self.cols)] for row in range(self.rows)
        ]
        self.mogs_right = [
            [self.cut_side(mogs[row][col], 'right') for col in range(self.cols)] for row in range(self.rows)
        ]
        self.mogs_top = [[self.cut_side(mogs[row][col], 'top') for col in range(self.cols)] for row in range(self.rows)]
        self.mogs_bottom = [
            [self.cut_side(mogs[row][col], 'bottom') for col in range(self.cols)] for row in range(self.rows)
        ]

    def load_grads(self):
        self.grads_x_left = [[0 for col in range(self.cols)] for row in range(self.rows)]
        self.grads_x_right = [[0 for col in range(self.cols)] for row in range(self.rows)]
        self.grads_x_top = [[0 for col in range(self.cols)] for row in range(self.rows)]
        self.grads_x_bottom = [[0 for col in range(self.cols)] for row in range(self.rows)]
        self.grads_y_left = [[0 for col in range(self.cols)] for row in range(self.rows)]
        self.grads_y_right = [[0 for col in range(self.cols)] for row in range(self.rows)]
        self.grads_y_top = [[0 for col in range(self.cols)] for row in range(self.rows)]
        self.grads_y_bottom = [[0 for col in range(self.cols)] for row in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.cols):
                grad_y, grad_x = np.gradient(self.frames[row][col])
                self.grads_x_left[row][col] = self.cut_side(grad_x, 'left')
                self.grads_x_right[row][col] = self.cut_side(grad_x, 'right')
                self.grads_x_top[row][col] = self.cut_side(grad_x, 'top')
                self.grads_x_bottom[row][col] = self.cut_side(grad_x, 'bottom')
                self.grads_y_left[row][col] = self.cut_side(grad_y, 'left')
                self.grads_y_right[row][col] = self.cut_side(grad_y, 'right')
                self.grads_y_top[row][col] = self.cut_side(grad_y, 'top')
                self.grads_y_bottom[row][col] = self.cut_side(grad_y, 'bottom')

    def delete_frames(self):
        self.frames = []
        self.frames_left = []
        self.frames_right = []
        self.frames_top = []
        self.frames_bottom = []

    def _comp_fun_median(self, im1, im2):
        # return self._thr_fun(im1) - self._thr_fun(im2)
        # (np.median(im1), np.median(im2)) * im1.size
        return np.sqrt(np.median(im1 - im2) ** 2) * im1.size

    def _difference_mean(self, im1, im2):
        return (im1 - im2).mean()

    def load_correlated_frames(self):
        # the order of frames is:
        # for frames_corr_R
        # 0 - 0
        # 1 - R
        # and
        # for frames_corr_B
        # 0 - 0
        # 1 - B
        frames_corr_R = [[[0 for _ in range(2)] for _ in range(self.cols)] for _ in range(self.rows)]
        frames_corr_B = copy.deepcopy(frames_corr_R)
        for row in range(self.rows):
            for col in range(self.cols):
                if col < self.cols - 1:
                    im0_R = self.cut_side(self.frames[row][col], 'right')
                    im_R = self.cut_side(self.frames[row][col + 1], 'left')
                    im0_R_crop, im_R_crop = correlate_images(im0_R, im_R)
                    frames_corr_R[row][col][0] = im0_R_crop
                    frames_corr_R[row][col][1] = im_R_crop
                if row < self.rows - 1:
                    im0_B = self.cut_side(self.frames[row][col], 'bottom')
                    im_B = self.cut_side(self.frames[row + 1][col], 'top')
                    im0_B_crop, im_B_crop = correlate_images(im0_B, im_B)
                    frames_corr_B[row][col][0] = im0_B_crop
                    frames_corr_B[row][col][1] = im_B_crop
        self.frames_corr_R = frames_corr_R
        self.frames_corr_B = frames_corr_B

    def offset_diffs_corr(self, col, row, offsets, comp_fun=None):
        if comp_fun is None:
            comp_fun = self._comp_fun
        idx0 = col + row * self.cols
        idx_R = col + 1 + row * self.cols
        idx_B = col + (row + 1) * self.cols
        diff_R = diff_B = 0
        if col < self.cols - 1:
            im0_R = self.frames_corr_R[row][col][0] + offsets[idx0]
            im_R = self.frames_corr_R[row][col][1] + offsets[idx_R]
            diff_R = comp_fun(im0_R, im_R)
        if row < self.rows - 1:
            im0_B = self.frames_corr_B[row][col][0] + offsets[idx0]
            im_B = self.frames_corr_B[row][col][1] + offsets[idx_B]
            diff_B = comp_fun(im0_B, im_B)
        return diff_R + diff_B

    def offset_diffs(self, col, row, offsets, maximize=False, comp_fun=None):
        if comp_fun is None:
            comp_fun = self._comp_fun
        idx = col + row * self.cols
        # idx +- self.cols is bottom/top neighbor
        # idx +- 1 is right/left neighbor
        im_0_left = self.frames_left[row][col].copy() + offsets[idx]
        im_0_right = self.frames_right[row][col].copy() + offsets[idx]
        im_0_top = self.frames_top[row][col].copy() + offsets[idx]
        im_0_bottom = self.frames_bottom[row][col].copy() + offsets[idx]
        diff_left = diff_right = diff_top = diff_bottom = 0
        if col > 0:
            # im_left = imread(str(self.paths[row][col - 1]))
            # im_left_cut = self.cut_side(im_left, 'right') + offsets[idx - 1]
            im_left_cut = self.frames_right[row][col - 1].copy() + offsets[idx - 1]
            # diff_left = rmse(im_0_left, im_left_cut) * im_left_cut.size
            diff_left = comp_fun(im_0_left, im_left_cut)
        if col < self.cols - 1:
            # im_right = imread(str(self.paths[row][col + 1]))
            # im_right_cut = self.cut_side(im_right, 'left') + offsets[idx + 1]
            im_right_cut = self.frames_left[row][col + 1].copy() + offsets[idx + 1]
            # diff_right = rmse(im_0_right, im_right_cut) * im_right_cut.size
            diff_right = comp_fun(im_0_right, im_right_cut)
        if row > 0:
            # im_top = imread(str(self.paths[row - 1][col]))
            # im_top_cut = self.cut_side(im_top, 'bottom') + offsets[idx - self.cols]
            im_top_cut = self.frames_bottom[row - 1][col].copy() + offsets[idx - self.cols]
            # diff_top = rmse(im_0_top, im_top_cut) * im_top_cut.size
            diff_top = comp_fun(im_0_top, im_top_cut)
        if row < self.rows - 1:
            # im_bottom = imread(str(self.paths[row + 1][col]))
            # im_bottom_cut = self.cut_side(im_bottom, 'top') + offsets[idx + self.cols]
            im_bottom_cut = self.frames_top[row + 1][col].copy() + offsets[idx + self.cols]
            # diff_bottom = rmse(im_0_bottom, im_bottom_cut) * im_bottom_cut.size
            diff_bottom = comp_fun(im_0_bottom, im_bottom_cut)
        diff_total = diff_left + diff_right + diff_top + diff_bottom
        if maximize is True:
            diff_total = diff_total * -1
        return diff_total

    def offset_diffs_init(self, col, row, maximize=False, comp_fun=None):
        def get_slopes(im, sigma=10):
            from scipy.ndimage import gaussian_filter, median_filter

            def filter_gradient(grad):
                grad_filt = gaussian_filter(grad, sigma=sigma)
                slope = np.median(grad_filt)
                return slope

            grad_y, grad_x = np.gradient(
                im, 2 / self.height, 2 / self.width
            )  # sampling rate – "2 / no_px" – due to -1:1 range in _calc_plane
            y_slope = filter_gradient(grad_y)
            x_slope = filter_gradient(grad_x)
            return x_slope, y_slope

        if comp_fun is None:
            comp_fun = self._comp_fun
        idx = col + row * self.cols
        # idx +- self.cols is bottom/top neighbor
        # idx +- 1 is right/left neighbor
        im_0_left = self.frames_left[row][col].copy()
        im_0_right = self.frames_right[row][col].copy()
        im_0_top = self.frames_top[row][col].copy()
        im_0_bottom = self.frames_bottom[row][col].copy()
        diff_left = diff_right = diff_top = diff_bottom = 0
        slope_diff_left_x = slope_diff_left_y = 0
        slope_diff_right_x = slope_diff_right_y = 0
        slope_diff_top_x = slope_diff_top_y = 0
        slope_diff_bottom_x = slope_diff_bottom_y = 0
        if col > 0:
            # im_left = imread(str(self.paths[row][col - 1]))
            # im_left_cut = self.cut_side(im_left, 'right') + offsets[idx - 1]
            im_left_cut = self.frames_right[row][col - 1].copy()
            # diff_left = rmse(im_0_left, im_left_cut) * im_left_cut.size
            diff_left = comp_fun(im_0_left, im_left_cut)
            im_diff_left = im_0_left - im_left_cut
            slope_diff_left_x, slope_diff_left_y = get_slopes(im_diff_left)
        if col < self.cols - 1:
            # im_right = imread(str(self.paths[row][col + 1]))
            # im_right_cut = self.cut_side(im_right, 'left') + offsets[idx + 1]
            im_right_cut = self.frames_left[row][col + 1].copy()
            # diff_right = rmse(im_0_right, im_right_cut) * im_right_cut.size
            diff_right = comp_fun(im_0_right, im_right_cut)
            im_diff_right = im_0_right - im_right_cut
            slope_diff_right_x, slope_diff_right_y = get_slopes(im_diff_right)
        if row > 0:
            # im_top = imread(str(self.paths[row - 1][col]))
            # im_top_cut = self.cut_side(im_top, 'bottom') + offsets[idx - self.cols]
            im_top_cut = self.frames_bottom[row - 1][col].copy()
            # diff_top = rmse(im_0_top, im_top_cut) * im_top_cut.size
            diff_top = comp_fun(im_0_top, im_top_cut)
            im_diff_top = im_0_top - im_top_cut
            slope_diff_top_x, slope_diff_top_y = get_slopes(im_diff_top)
        if row < self.rows - 1:
            # im_bottom = imread(str(self.paths[row + 1][col]))
            # im_bottom_cut = self.cut_side(im_bottom, 'top') + offsets[idx + self.cols]
            im_bottom_cut = self.frames_top[row + 1][col].copy()
            # diff_bottom = rmse(im_0_bottom, im_bottom_cut) * im_bottom_cut.size
            diff_bottom = comp_fun(im_0_bottom, im_bottom_cut)
            im_diff_bottom = im_0_bottom - im_bottom_cut
            slope_diff_bottom_x, slope_diff_bottom_y = get_slopes(im_diff_bottom)
        # diff_total = diff_left + diff_right + diff_top + diff_bottom
        test = 1
        return {
            'L': diff_left / 2,
            'R': diff_right / 2,
            'T': diff_top / 2,
            'B': diff_bottom / 2,
            'L_x': slope_diff_left_x / 2,
            'L_y': slope_diff_left_y / 2,
            'R_x': slope_diff_right_x / 2,
            'R_y': slope_diff_right_y / 2,
            'T_x': slope_diff_top_x / 2,
            'T_y': slope_diff_top_y / 2,
            'B_x': slope_diff_bottom_x / 2,
            'B_y': slope_diff_bottom_y / 2,
        }

    def offset_diffs_separate(self, col, row, offsets, slopes=None, add_slopes_to_offsets=False, comp_fun=None):
        if comp_fun is None:
            comp_fun = self._comp_fun
        idx = col + row * self.cols
        # idx +- self.cols is bottom/top neighbor
        # idx +- 1 is right/left neighbor
        im_0_left = self.frames_summary[row][col]['L'] + offsets[idx]
        im_0_right = self.frames_summary[row][col]['R'] + offsets[idx]
        im_0_top = self.frames_summary[row][col]['T'] + offsets[idx]
        im_0_bottom = self.frames_summary[row][col]['B'] + offsets[idx]
        if slopes is not None:
            if add_slopes_to_offsets:
                center_of_overlap = 1 - self.overlap / 2
                im_0_left -= center_of_overlap * slopes[idx][0]
                im_0_right += center_of_overlap * slopes[idx][0]
                im_0_top -= center_of_overlap * slopes[idx][1]
                im_0_bottom += center_of_overlap * slopes[idx][1]
            im_0_left_x = self.frames_summary[row][col]['L_x'] + slopes[idx][0]
            im_0_right_x = self.frames_summary[row][col]['R_x'] + slopes[idx][0]
            im_0_top_x = self.frames_summary[row][col]['T_x'] + slopes[idx][0]
            im_0_bottom_x = self.frames_summary[row][col]['B_x'] + slopes[idx][0]
            im_0_left_y = self.frames_summary[row][col]['L_y'] + slopes[idx][1]
            im_0_right_y = self.frames_summary[row][col]['R_y'] + slopes[idx][1]
            im_0_top_y = self.frames_summary[row][col]['T_y'] + slopes[idx][1]
            im_0_bottom_y = self.frames_summary[row][col]['B_y'] + slopes[idx][1]
        diff_left = diff_right = diff_top = diff_bottom = 0
        slope_diff_left_x = slope_diff_left_y = 0
        slope_diff_right_x = slope_diff_right_y = 0
        slope_diff_top_x = slope_diff_top_y = 0
        slope_diff_bottom_x = slope_diff_bottom_y = 0
        if col > 0:
            # im_left = imread(str(self.paths[row][col - 1]))
            # im_left_cut = self.cut_side(im_left, 'right') + offsets[idx - 1]
            im_left_cut = self.frames_summary[row][col - 1]['R'] + offsets[idx - 1]
            if slopes is not None and add_slopes_to_offsets:
                im_left_cut += center_of_overlap * slopes[idx - 1][0]
            # diff_left = rmse(im_0_left, im_left_cut) * im_left_cut.size
            diff_left = comp_fun(im_0_left, im_left_cut)
            if slopes is not None:
                slope_left_x = self.frames_summary[row][col - 1]['R_x'] + slopes[idx - 1][0]
                slope_diff_left_x = im_0_left_x - slope_left_x
                slope_left_y = self.frames_summary[row][col - 1]['R_y'] + slopes[idx - 1][1]
                slope_diff_left_y = im_0_left_y - slope_left_y
        if col < self.cols - 1:
            # im_right = imread(str(self.paths[row][col + 1]))
            # im_right_cut = self.cut_side(im_right, 'left') + offsets[idx + 1]
            im_right_cut = self.frames_summary[row][col + 1]['L'] + offsets[idx + 1]
            if slopes is not None and add_slopes_to_offsets:
                im_right_cut -= center_of_overlap * slopes[idx + 1][0]
            # diff_right = rmse(im_0_right, im_right_cut) * im_right_cut.size
            diff_right = comp_fun(im_0_right, im_right_cut)
            if slopes is not None:
                slope_right_x = self.frames_summary[row][col - 1]['L_x'] + slopes[idx + 1][0]
                slope_diff_right_x = im_0_right_x - slope_right_x
                slope_right_y = self.frames_summary[row][col - 1]['L_y'] + slopes[idx + 1][1]
                slope_diff_right_y = im_0_right_y - slope_right_y
        if row > 0:
            # im_top = imread(str(self.paths[row - 1][col]))
            # im_top_cut = self.cut_side(im_top, 'bottom') + offsets[idx - self.cols]
            im_top_cut = self.frames_summary[row - 1][col]['B'] + offsets[idx - self.cols]
            if slopes is not None and add_slopes_to_offsets:
                im_top_cut += center_of_overlap * slopes[idx - self.cols][1]
            # diff_top = rmse(im_0_top, im_top_cut) * im_top_cut.size
            diff_top = comp_fun(im_0_top, im_top_cut)
            if slopes is not None:
                slope_top_x = self.frames_summary[row][col - 1]['B_x'] + slopes[idx - self.cols][0]
                slope_diff_top_x = im_0_top_x - slope_top_x
                slope_top_y = self.frames_summary[row][col - 1]['B_y'] + slopes[idx - self.cols][1]
                slope_diff_top_y = im_0_top_y - slope_top_y
        if row < self.rows - 1:
            # im_bottom = imread(str(self.paths[row + 1][col]))
            # im_bottom_cut = self.cut_side(im_bottom, 'top') + offsets[idx + self.cols]
            im_bottom_cut = self.frames_summary[row + 1][col]['T'] + offsets[idx + self.cols]
            if slopes is not None and add_slopes_to_offsets:
                im_bottom_cut -= center_of_overlap * slopes[idx + self.cols][1]
            # diff_bottom = rmse(im_0_bottom, im_bottom_cut) * im_bottom_cut.size
            diff_bottom = comp_fun(im_0_bottom, im_bottom_cut)
            if slopes is not None:
                slope_bottom_x = self.frames_summary[row][col - 1]['T_x'] + slopes[idx + self.cols][0]
                slope_diff_bottom_x = im_0_bottom_x - slope_bottom_x
                slope_bottom_y = self.frames_summary[row][col - 1]['T_y'] + slopes[idx + self.cols][1]
                slope_diff_bottom_y = im_0_bottom_y - slope_bottom_y
        diff_mean = np.nanmean(np.array([diff_left, diff_right, diff_top, diff_bottom])) / 2
        diff_x_mean = (
            np.nanmean(np.array([slope_diff_left_x, slope_diff_right_x, slope_diff_top_x, slope_diff_bottom_x])) / 2
        )
        diff_y_mean = (
            np.nanmean(np.array([slope_diff_left_y, slope_diff_right_y, slope_diff_top_y, slope_diff_bottom_y])) / 2
        )
        if slopes is not None:
            return diff_mean, diff_x_mean, diff_y_mean
        else:
            return diff_mean

    def minimize_offsets_error_corr_iterative(self, offsets, iterations, downsample=1, lr=1, amp=1, comp_fun=None):
        if comp_fun is None:
            comp_fun = lambda x, y: np.median(x - y)
        if not hasattr(self, 'frames'):
            self.load_frames(downsample_rate=downsample)
        if not hasattr(self, 'frames_corr_R'):
            self.load_correlated_frames()
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        self.x, self.y = np.meshgrid(x, y)
        for i in range(iterations):
            print(f'Iteration: {i}/{iterations}')
            offsets_corr = np.zeros_like(self.offsets)
            for row in range(self.rows):
                for col in range(self.cols):
                    idx = col + row * self.cols
                    corr = (
                        self.offset_diffs_corr(col, row, offsets, comp_fun=comp_fun) / 4 * lr
                    )  # 2 to meet halfway for each component, 2 components: right, bottom
                    if np.isnan(corr):
                        test = 1
                    offsets_corr[idx] = corr
            offsets -= offsets_corr
            lr *= amp

    def minimize_offsets_error_iterative(self, offsets, iterations, downsample=1, lr=1, amp=1):
        comp_fun_local_l = lambda x, y: np.mean(x - y)
        # comp_fun_local_l = lambda x, y: np.median(np.abs(x-y))
        def comp_fun_local(im1, im2):
            diff = im1 - im2
            return np.mean((np.sqrt((im1 - im2) ** 2) * np.sign(diff)))

        def comp_fun_local_2(im1, im2):
            min1, max1 = im1.min(), im1.max()
            min2, max2 = im2.min(), im2.max()
            diff_min = min1 - min2
            diff_max = max1 - max2
            return (diff_min + diff_max) / 2

        def comp_fun_local_3(im1, im2):
            diff = im1 - im2
            shifts, rmse, _ = register_translation(im1, im2, upsample_factor=1)
            return rmse * np.sign(np.mean(diff))

        # if not hasattr(self, 'frames'):
        # self.load_frames(downsample_rate=downsample)
        self.downsample = downsample
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        self.x, self.y = np.meshgrid(x, y)
        if downsample > 1:
            factors = (downsample, downsample)
            temp_frame = downscale_local_mean(self.frames[0][0], factors)
            y = np.linspace(-1, 1, temp_frame.shape[0])
            x = np.linspace(-1, 1, temp_frame.shape[1])
            self.x_ds, self.y_ds = np.meshgrid(x, y)
        for i in range(iterations):
            print(f'Iteration: {i}/{iterations}')
            offsets_corr = np.zeros_like(self.offsets)
            for row in range(self.rows):
                for col in range(self.cols):
                    idx = col + row * self.cols
                    # corr = self.offset_diffs(col, row, offsets, comp_fun=self._difference_mean) / 8 # 2 to meet halfway for each component, 4 components: left, right, top, down
                    corr = (
                        self.offset_diffs(col, row, offsets, comp_fun=comp_fun_local_l) / 8 * lr
                    )  # 2 to meet halfway for each component, 4 components: left, right, top, down
                    if np.isnan(corr):
                        test = 1
                    offsets_corr[idx] = corr
            offsets -= offsets_corr
            lr *= amp
        offsets -= offsets.mean()

    def minimize_offsets_error_iterative_summarized(self, offsets, iterations, downsample=1, lr=1, amp=1):
        # Operates only on summarized (eg. mean) differences, not full images (faster)
        comp_fun_local_l = lambda x, y: np.mean(x - y)
        # comp_fun_local_l = lambda x, y: np.median(np.abs(x - y))

        def comp_fun_local(im1, im2):
            diff = im1 - im2
            return np.mean((np.sqrt((im1 - im2) ** 2) * np.sign(diff)))

        def comp_fun_local_2(im1, im2):
            min1, max1 = im1.min(), im1.max()
            min2, max2 = im2.min(), im2.max()
            diff_min = min1 - min2
            diff_max = max1 - max2
            return (diff_min + diff_max) / 2

        def comp_fun_local_3(im1, im2):
            diff = im1 - im2
            shifts, rmse, _ = register_translation(im1, im2, upsample_factor=1)
            return rmse * np.sign(np.mean(diff))

        # if not hasattr(self, 'frames'):
        # self.load_frames(downsample_rate=downsample)
        self.downsample = downsample
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        self.x, self.y = np.meshgrid(x, y)
        if downsample > 1:
            factors = (downsample, downsample)
            temp_frame = downscale_local_mean(self.frames[0][0], factors)
            y = np.linspace(-1, 1, temp_frame.shape[0])
            x = np.linspace(-1, 1, temp_frame.shape[1])
            self.x_ds, self.y_ds = np.meshgrid(x, y)

        self.frames_summary = [
            [self.offset_diffs_init(col, row, offsets, comp_fun=comp_fun_local_l) for col in range(self.cols)]
            for row in range(self.rows)
        ]

        # offsets_corr = np.zeros_like(self.offsets)
        # for row in range(self.rows):
        #     for col in range(self.cols):
        #         idx = col + row * self.cols
        #         corr = [self.frames_summary[row][col][key] for key in self.frames_summary[row][col]]
        #         corr = np.array(corr).mean()
        #         offsets_corr[idx] = corr

        for i in range(iterations):
            print(f'Iteration: {i}/{iterations}')
            offsets_corr = np.zeros_like(self.offsets)
            for row in range(self.rows):
                for col in range(self.cols):
                    idx = col + row * self.cols
                    # corr = self.offset_diffs(col, row, offsets, comp_fun=self._difference_mean) / 8 # 2 to meet halfway for each component, 4 components: left, right, top, down
                    corr = self.offset_diffs_separate(col, row, offsets, comp_fun=comp_fun_local_l) * lr
                    if np.isnan(corr):
                        test = 1
                    offsets_corr[idx] = corr
            offsets -= offsets_corr
            lr *= amp
        offsets -= offsets.mean()

    def minimize_offsets_tilts_error_iterative_summarized(
        self, planes_params, iterations, downsample=1, lr=1, amp=1, lr_tilt=1, amp_tilt=1
    ):
        comp_fun_local_l = lambda x, y: np.mean(x - y)
        # comp_fun_local_l = lambda x, y: np.median(np.abs(x - y))

        offsets = np.array([planes_params[i * 3] for i in range(planes_params.shape[0] // 3)])
        slopes = np.array(
            [(planes_params[i * 3 + 1], planes_params[i * 3 + 2]) for i in range(planes_params.shape[0] // 3)]
        )

        # if not hasattr(self, 'frames'):
        # self.load_frames(downsample_rate=downsample)
        self.downsample = downsample
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        self.x, self.y = np.meshgrid(x, y)
        if downsample > 1:
            factors = (downsample, downsample)
            temp_frame = downscale_local_mean(self.frames[0][0], factors)
            y = np.linspace(-1, 1, temp_frame.shape[0])
            x = np.linspace(-1, 1, temp_frame.shape[1])
            self.x_ds, self.y_ds = np.meshgrid(x, y)

        self.frames_summary = [
            [self.offset_diffs_init(col, row, offsets, comp_fun=comp_fun_local_l) for col in range(self.cols)]
            for row in range(self.rows)
        ]

        # planes_corr = np.zeros_like(self.planes_params)
        # for row in range(self.rows):
        #     for col in range(self.cols):
        #         idx3 = (col + row * self.cols) * 3  # 3 times as many parameters as frames (3 params per frame)
        #         corr = [self.frames_summary[row][col][key] for key in ['L', 'R', 'T', 'B']]
        #         corr_x = [self.frames_summary[row][col][key] for key in ['L_x', 'R_x', 'T_x', 'B_x']]
        #         corr_y = [self.frames_summary[row][col][key] for key in ['L_y', 'R_y', 'T_y', 'B_y']]
        #         corr = np.array(corr).mean()
        #         corr_x = np.array(corr_x).mean()
        #         corr_y = np.array(corr_y).mean()
        #         planes_corr[idx3] = corr
        #         planes_corr[idx3 + 1] = corr_x
        #         planes_corr[idx3 + 2] = corr_y

        for i in range(iterations):
            print(f'Iteration: {i}/{iterations}')

            offsets_corr = np.zeros_like(self.offsets)
            # offsets_shape = offsets_corr.shape
            slopes_corr = [[0, 0] for _ in range(self.offsets.shape[0])]
            for row in range(self.rows):
                for col in range(self.cols):
                    idx = col + row * self.cols
                    # corr = self.offset_diffs(col, row, offsets, comp_fun=self._difference_mean) / 8 # 2 to meet halfway for each component, 4 components: left, right, top, down
                    corr, corr_x, corr_y = self.offset_diffs_separate(
                        col, row, offsets, slopes, comp_fun=comp_fun_local_l
                    )
                    if np.isnan(corr):
                        test = 1
                    offsets_corr[idx] = corr * lr
                    slopes_corr[idx][0] = corr_x * lr_tilt
                    slopes_corr[idx][1] = corr_y * lr_tilt
                    slopes[idx][0] -= corr_x
                    slopes[idx][1] -= corr_y
            offsets -= offsets_corr
            # slopes[0] -= slopes_corr[0]
            # slopes[1] -= slopes_corr[1]
            lr *= amp
            lr_tilt *= amp_tilt
        offsets -= offsets.mean()
        planes_params = np.zeros_like(planes_params)
        for i in range(offsets.shape[0]):
            planes_params[i * 3] = offsets[i]
            planes_params[i * 3 + 1] = slopes[i][0]
            planes_params[i * 3 + 2] = slopes[i][1]
        self.planes_params = planes_params
        # self.offsets = offsets

    def offsets_fitness(self, offsets, maximize=False):
        fitness = 0
        for row in range(self.rows):
            for col in range(self.cols):
                # idx = col + row * self.cols
                fitness += self.offset_diffs(
                    col, row, offsets, maximize, comp_fun=(lambda x, y: np.mean(np.abs(x - y)))
                )
        if self.iter_count % 100 == 0:
            print(f'Optimization iterations: {self.iter_count}')
        self.iter_count += 1
        return fitness

    def minimize_offsets_error(self, maxiter=100, downsample=1, comp_fun=None):
        # self.load_frames(downsample_rate=downsample)
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        self.x, self.y = np.meshgrid(x, y)
        self.downsample = downsample
        if downsample > 1:
            factors = (downsample, downsample)
            temp_frame = downscale_local_mean(self.frames[0][0], factors)
            y = np.linspace(-1, 1, temp_frame.shape[0])
            x = np.linspace(-1, 1, temp_frame.shape[1])
            self.x_ds, self.y_ds = np.meshgrid(x, y)
        self.downsample = downsample
        self.iter_count = 0
        res = minimize(self.offsets_fitness, self.offsets, options={'maxiter': maxiter})
        self.offsets = res.x
        return res

    def variation_fun(self, im1, im2):
        return np.sqrt(np.asarray([np.sum(im1**2), np.sum(im2**2)]).sum())

    def _thr_fun(self, im):
        thr = threshold_otsu(im)
        w = 0
        w += im[im >= thr].mean() * im[im >= thr].size
        w += im[im < thr].mean() * im[im < thr].size
        return w

    def _comp_fun(self, im1, im2):
        # return self._thr_fun(im1) - self._thr_fun(im2)
        # return rmse(im1, im2) * im1.size
        return np.sqrt((im1 - im2) ** 2).sum()

    def _comp_fun_2(self, im1, im2):
        # return self._thr_fun(im1) - self._thr_fun(im2)
        # (np.median(im1), np.median(im2)) * im1.size
        return np.sqrt(np.median(im1 - im2) ** 2) * im1.size

    def plane_fitting_diffs_3(self, col, row, lam_edges=0.0001, lam_tv=0):
        # idx = col + row * self.cols
        # idx +- self.cols is bottom/top neighbor
        # idx +- 1 is right/left neighbor
        im_0_left = self.frames_left[row][col] + self.planes_left[row][col]
        im_0_right = self.frames_right[row][col] + self.planes_right[row][col]
        im_0_top = self.frames_top[row][col] + self.planes_top[row][col]
        im_0_bottom = self.frames_bottom[row][col] + self.planes_bottom[row][col]
        diff_left = diff_right = diff_top = diff_bottom = 0
        # metric = (im_0_left.mean() + im_0_right.mean() + im_0_top.mean() + im_0_bottom.mean()) * 4 * self.width * self.height * self.overlap
        metric = 0
        if lam_tv > 0:
            # metric += self._comp_fun(im_0_left, im_0_left.mean())
            # metric += self._comp_fun(im_0_right, im_0_right.mean())
            # metric += self._comp_fun(im_0_top, im_0_top.mean())
            # metric += self._comp_fun(im_0_bottom, im_0_bottom.mean())
            # metric += self._comp_fun(tv(im_0_left), 0)
            # metric += self._comp_fun(tv(im_0_right), 0)
            # metric += self._comp_fun(tv(im_0_top), 0)
            # metric += self._comp_fun(tv(im_0_bottom), 0)
            metric += tv(im_0_left)
            metric += tv(im_0_right)
            metric += tv(im_0_top)
            metric += tv(im_0_bottom)
            # if col > 0:
            #     im_left_cut = self.frames_right[row][col - 1] + self.planes_right[row][col - 1]
            #     metric += tv(im_0_left - im_left_cut)
            # if col < self.cols - 1:
            #     im_right_cut = self.frames_left[row][col + 1] + self.planes_left[row][col + 1]
            #     metric += tv(im_0_right - im_right_cut)
            # if row > 0:
            #     im_top_cut = self.frames_bottom[row - 1][col] + self.planes_bottom[row - 1][col]
            #     metric += tv(im_0_top - im_top_cut)
            # if row < self.rows - 1:
            #     im_bottom_cut = self.frames_top[row + 1][col] + self.planes_top[row + 1][col]
            #     metric += tv(im_0_bottom - im_bottom_cut)
        if col > 0:
            im_left_cut = self.frames_right[row][col - 1] + self.planes_right[row][col - 1]
            diff_left = self._comp_fun(im_0_left, im_left_cut)
        if col < self.cols - 1:
            im_right_cut = self.frames_left[row][col + 1] + self.planes_left[row][col + 1]
            diff_right = self._comp_fun(im_0_right, im_right_cut)
        if row > 0:
            im_top_cut = self.frames_bottom[row - 1][col] + self.planes_bottom[row - 1][col]
            diff_top = self._comp_fun(im_0_top, im_top_cut)
        if row < self.rows - 1:
            im_bottom_cut = self.frames_top[row + 1][col] + self.planes_top[row + 1][col]
            diff_bottom = self._comp_fun(im_0_bottom, im_bottom_cut)
        if lam_edges > 0:
            lam = lam_edges
            if col == 0:
                diff_left = self._comp_fun(im_0_left, im_0_left.mean())  # rmse(im_0_left, 0) * im_0_left.size * lam
            if col == self.cols - 1:
                diff_right = self._comp_fun(
                    im_0_right, im_0_right.mean()
                )  # rmse(im_0_right, 0) * im_0_right.size * lam
            if row == 0:
                diff_top = self._comp_fun(im_0_top, im_0_top.mean())  # rmse(im_0_top, 0) * im_0_top.size * lam
            if row == self.rows - 1:
                diff_bottom = self._comp_fun(
                    im_0_bottom, im_0_bottom.mean()
                )  # rmse(im_0_bottom, 0) * im_0_bottom.size * lam
        diff_total = diff_left + diff_right + diff_top + diff_bottom
        return diff_total, metric

    # @jit
    def plane_fitting_diffs_cols(self, col, row, edges=True, lam_edges=0.0001):
        col_left = col - 1
        col_right = col
        diff = im_left = im_right = 0
        if row < self.rows:
            if col > 0 and col < self.cols:
                im_left = self.frames_left[row][col_left] + self.planes_left[row][col_left]
                im_right = self.frames_right[row][col_right] + self.planes_right[row][col_right]
                diff = rmse(im_left, im_right) * im_left.size
            else:
                if edges:
                    if col == 0:
                        im_right = self.frames_right[row][col_right] + self.planes_right[row][col_right]
                        im_size = im_right.size
                        im_left = im_right.mean()
                    if col == self.cols:
                        im_left = self.frames_left[row][col_left] + self.planes_left[row][col_left]
                        im_size = im_left.size
                        im_right = im_left.mean()
                    diff = rmse(im_left, im_right) * im_size * lam_edges
        im_tv = self.variation_fun(im_left, im_right)
        return diff, im_tv

    # @jit
    def plane_fitting_diffs_rows(self, col, row, edges=True, lam_edges=0.0001):
        row_top = row - 1
        row_bottom = row
        diff = im_top = im_bottom = 0
        if col < self.cols:
            if row > 0 and row < self.rows:
                im_top = self.frames_top[row_top][col] + self.planes_top[row_top][col]
                im_bottom = self.frames_bottom[row_bottom][col] + self.planes_bottom[row_bottom][col]
                diff = rmse(im_top, im_bottom) * im_top.size
            else:
                if edges:
                    if row == 0:
                        im_bottom = self.frames_bottom[row_bottom][col] + self.planes_bottom[row_bottom][col]
                        im_size = im_bottom.size
                        im_top = im_bottom.mean()
                    if row == self.rows:
                        im_top = self.frames_top[row_top][col] + self.planes_top[row_top][col]
                        im_size = im_top.size
                        im_bottom = im_top.mean()
                    diff = rmse(im_top, im_bottom) * im_size * lam_edges
        im_tv = self.variation_fun(im_top, im_bottom)
        return diff, im_tv

    def _calc_plane(self, c, override_downsample=False):
        if not hasattr(self, 'downsample'):
            self.downsample = 1
        if self.downsample == 1 or override_downsample:
            if not hasattr(self, 'x'):
                y = np.linspace(-1, 1, self.height)
                x = np.linspace(-1, 1, self.width)
                self.x, self.y = np.meshgrid(x, y)
            return c[0] + self.x * c[1] + self.y * c[2]
        else:
            if not hasattr(self, 'x_ds'):
                factors = (self.downsample, self.downsample)
                temp_frame = downscale_local_mean(imread(str(self.paths[0][0])), factors)
                y = np.linspace(-1, 1, temp_frame.shape[0])
                x = np.linspace(-1, 1, temp_frame.shape[1])
                self.x_ds, self.y_ds = np.meshgrid(x, y)
            return c[0] + self.x_ds * c[1] + self.y_ds * c[2]

    def zero_planes_params(self, offsets=True, slope_x=True, slope_y=True):
        for row in range(self.rows):
            for col in range(self.cols):
                idx3 = (col + row * self.cols) * 3
                if offsets:
                    self.planes_params[idx3] = 0
                if slope_x:
                    self.planes_params[idx3 + 1] = 0
                if slope_y:
                    self.planes_params[idx3 + 2] = 0

    def extract_from_planes_params(self, position=0):
        param = np.zeros(self.rows * self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                idx3 = (col + row * self.cols) * 3
                idx = col + row * self.cols
                param[idx] = self.planes_params[idx3 + position]
        return param

    def add_to_planes_params(self, param, position=0):
        for row in range(self.rows):
            for col in range(self.cols):
                idx3 = (col + row * self.cols) * 3
                idx = col + row * self.cols
                self.planes_params[idx3 + position] = param[idx]
        return param

    def filter_planes_params(self, filter_fun=None, filter_offsets=False, filter_slopes=True):
        if filter_fun is None:
            from scipy.ndimage import gaussian_filter

            filter_fun = lambda x: gaussian_filter(x, sigma=1.25)
        offsets = np.zeros(self.rows * self.cols)
        slopes_x = np.zeros_like(offsets)
        slopes_y = np.zeros_like(offsets)
        for i in range(self.rows * self.cols):
            offsets[i] = self.planes_params[i * 3]
            slopes_x[i] = self.planes_params[i * 3 + 1]
            slopes_y[i] = self.planes_params[i * 3 + 2]
        offsets_im = offsets.copy().reshape((self.rows, self.cols))
        slopes_x_im = slopes_x.copy().reshape((self.rows, self.cols))
        slopes_y_im = slopes_y.copy().reshape((self.rows, self.cols))
        # slopes_x_lf, _ = legendre_fitting_nan(slopes_x_im, 4)
        # slopes_y_lf, _ = legendre_fitting_nan(slopes_y_im, 4)
        offsets_im_lf = filter_fun(offsets_im)
        slopes_x_lf = filter_fun(slopes_x_im)
        slopes_y_lf = filter_fun(slopes_y_im)
        offsets_im_hf = offsets_im - offsets_im_lf
        slopes_x_hf = slopes_x_im - slopes_x_lf
        slopes_y_hf = slopes_y_im - slopes_y_lf
        if filter_offsets:
            offsets = offsets_im_hf.copy().reshape(-1)
        if filter_slopes:
            slopes_x = slopes_x_hf.copy().reshape(-1)
            slopes_y = slopes_y_hf.copy().reshape(-1)
        for i in range(self.rows * self.cols):
            self.planes_params[i * 3] = offsets[i]
            self.planes_params[i * 3 + 1] = slopes_x[i]
            self.planes_params[i * 3 + 2] = slopes_y[i]

    def subtract_from_offsets_in_planes_params(self, a=None):
        if a is None:
            a = []
            for i in range(self.rows * self.cols):
                a.append(self.planes_params[i * 3])
            a = np.asarray(a).mean()
        for i in range(self.rows * self.cols):
            self.planes_params[i * 3] -= a

    def _calc_planes(self, planes_params):
        for row in range(self.rows):
            for col in range(self.cols):
                idx3 = (col + row * self.cols) * 3  # 3 times as many parameters as frames (3 params per frame)
                plane_params = (planes_params[idx3], planes_params[idx3 + 1], planes_params[idx3 + 2])
                plane = self._calc_plane(plane_params)
                self.planes_left[row][col] = self.cut_side(plane, 'left')
                self.planes_right[row][col] = self.cut_side(plane, 'right')
                self.planes_top[row][col] = self.cut_side(plane, 'top')
                self.planes_bottom[row][col] = self.cut_side(plane, 'bottom')

    def _calc_grads(self):
        from scipy.ndimage import gaussian_filter, median_filter

        def filter_gradient(grad):
            grad_filt = gaussian_filter(grad, sigma=10)
            slope = np.median(grad_filt)
            return slope

        self.init_grads = np.zeros(self.rows * self.cols * 2)
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                grad_y, grad_x = np.gradient(im)
                idx2 = (col + row * self.cols) * 2
                # x_0 = grad_x.sum() / self.width / self.height # normalized, simply add slopes
                # y_0 = grad_y.sum() / self.width / self.height # normalized, simply add slopes
                x_0 = filter_gradient(grad_x)
                y_0 = filter_gradient(grad_y)
                self.init_grads[idx2] = x_0
                self.init_grads[idx2 + 1] = y_0

    def plane_fitting_fitness_grad_regularized(self, planes_params, lam_edges, lam_tv):
        self._calc_planes(planes_params)
        if self.iter_count == 0:
            self._calc_grads()
        fitness = 0
        im_tv = 0
        for row in range(self.rows):
            for col in range(self.cols):
                fit, _ = self.plane_fitting_diffs_3(col, row, lam_edges=0, lam_tv=0)
                fitness += fit
                if lam_tv > 0:
                    idx2 = (col + row * self.cols) * 2
                    idx3 = (col + row * self.cols) * 3
                    x_grad = self.init_grads[idx2] + planes_params[idx3 + 1]
                    y_grad = self.init_grads[idx2 + 1] + planes_params[idx3 + 2]
                    im_tv += np.sqrt(x_grad**2 + y_grad**2)
        tv_full = im_tv * self.overlap * self.height * self.width
        if self.iter_count % 100 == 0:
            print(f'Optimization iterations: {self.iter_count}')
        self.iter_count += 1
        return fitness + tv_full * lam_tv

    def plane_fitting_fitness(self, planes_params, lam_edges, lam_tv):
        self._calc_planes(planes_params)
        fitness = 0
        im_tv = 0
        # tv_list = []
        for row in range(self.rows):
            for col in range(self.cols):
                # fitness += self.plane_fitting_diffs(col, row, maximize, edges=True)
                fit, _ = self.plane_fitting_diffs_3(col, row, lam_edges=lam_edges)
                if lam_tv > 0:
                    im_tv += self.sides_tv_2(col, row, mean=True)
                fitness += fit
                # tv_list.append(im_tv)
        # tv_full = np.asarray(tv_list).sum()
        if self.iter_count % 100 == 0:
            print(f'Optimization iterations: {self.iter_count}')
        self.iter_count += 1
        return fitness + im_tv * lam_tv

    # @jit
    def plane_fitting_fitness_2(self, planes_params, lam_edges, lam_tv):
        self._calc_planes(planes_params)
        fitness = 0
        tv_list = []
        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                fit, im_tv = self.plane_fitting_diffs_cols(col, row, lam_edges=lam_edges)
                fitness += fit
                tv_list.append(im_tv)
                fit, im_tv = self.plane_fitting_diffs_rows(col, row, lam_edges=lam_edges)
                fitness += fit
                tv_list.append(im_tv)
        tv_full = np.asarray(tv_list).sum() * 0.5
        return fitness + tv_full * lam_tv

    def plane_fitting_fitness_skew(self, planes_params, lam_edges, lam_skew):
        self._calc_planes(planes_params)
        fitness = 0
        for row in range(self.rows):
            for col in range(self.cols):
                # fitness += self.plane_fitting_diffs(col, row, maximize, edges=True)
                fit, _ = self.plane_fitting_diffs_3(col, row, lam_edges=lam_edges)
                fit = fit / (self.skewness(col, row, planes_params) * lam_skew + 1)
                # fit += self.skewness(col, row, planes_params) * lam_skew
                fitness = fitness + fit
        return fitness

    def skewness(self, col, row, planes_params):
        idx3 = (col + row * self.cols) * 3
        x_0 = planes_params[idx3 + 1]
        y_0 = planes_params[idx3 + 2]
        return np.sqrt(x_0**2 + y_0**2)

    def minimize_plane_fitting_error(self, f, maxiter=100, lam_edges=0.0001, lam_tv=0.0005, downsample=1):
        # self.load_frames(downsample_rate=downsample)
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        self.x, self.y = np.meshgrid(x, y)
        self.downsample = downsample
        if downsample > 1:
            factors = (downsample, downsample)
            temp_frame = downscale_local_mean(self.frames[0][0], factors)
            y = np.linspace(-1, 1, temp_frame.shape[0])
            x = np.linspace(-1, 1, temp_frame.shape[1])
            self.x_ds, self.y_ds = np.meshgrid(x, y)
        # self.load_grads()
        self.iter_count = 0
        res = minimize(f, self.planes_params, args=(lam_edges, lam_tv), options={'maxiter': maxiter})
        self.planes_params = res.x
        return res

    def minimize_plane_fitting_error_2(self, maxiter=100, lam_edges=0.0001, lam_tv=0.0005):
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        self.x, self.y = np.meshgrid(x, y)
        # self.load_frames()
        res = minimize(
            self.plane_fitting_fitness_2, self.planes_params, args=(lam_edges, lam_tv), options={'maxiter': maxiter}
        )
        self.planes_params = res.x
        return res

    def sides_tv(self, col, row, planes_params, lam_edges=0.001, maximize=False, edges=True):
        # idx = col + row * self.cols
        # idx +- self.cols is bottom/top neighbor
        # idx +- 1 is right/left neighbor
        idx3 = (col + row * self.cols) * 3
        x_0 = planes_params[idx3 + 1]
        y_0 = planes_params[idx3 + 2]
        im_0_x_left = self.grads_x_left[row][col] + x_0
        im_0_x_right = self.grads_x_right[row][col] + x_0
        im_0_x_top = self.grads_x_top[row][col] + x_0
        im_0_x_bottom = self.grads_x_bottom[row][col] + x_0
        im_0_y_left = self.grads_y_left[row][col] + y_0
        im_0_y_right = self.grads_y_right[row][col] + y_0
        im_0_y_top = self.grads_y_top[row][col] + y_0
        im_0_y_bottom = self.grads_y_bottom[row][col] + y_0
        diff_left = diff_right = diff_top = diff_bottom = 0
        if col > 0:
            idx3_left = (col - 1 + row * self.cols) * 3
            x_left = planes_params[idx3_left + 1]
            y_left = planes_params[idx3_left + 2]
            im_left_cut_x = self.grads_x_right[row][col - 1] + x_left
            im_left_cut_y = self.grads_y_right[row][col - 1] + y_left
            diff_left += mse(im_0_x_left, im_left_cut_x) * im_left_cut_x.size
            diff_left += mse(im_0_y_left, im_left_cut_y) * im_left_cut_y.size
        if col < self.cols - 1:
            idx3_right = (col + 1 + row * self.cols) * 3
            x_right = planes_params[idx3_right + 1]
            y_right = planes_params[idx3_right + 2]
            im_right_cut_x = self.grads_x_left[row][col + 1] + x_right
            im_right_cut_y = self.grads_y_left[row][col + 1] + y_right
            diff_right += mse(im_0_x_right, im_right_cut_x) * im_right_cut_x.size
            diff_right += mse(im_0_y_right, im_right_cut_y) * im_right_cut_y.size
        if row > 0:
            idx3_top = (col + (row - 1) * self.cols) * 3
            x_top = planes_params[idx3_top + 1]
            y_top = planes_params[idx3_top + 2]
            im_top_cut_x = self.grads_x_bottom[row - 1][col] + x_top
            im_top_cut_y = self.grads_y_bottom[row - 1][col] + y_top
            diff_top += mse(im_0_x_top, im_top_cut_x) * im_top_cut_x.size
            diff_top += mse(im_0_y_top, im_top_cut_y) * im_top_cut_y.size
        if row < self.rows - 1:
            idx3_bottom = (col + (row + 1) * self.cols) * 3
            x_bottom = planes_params[idx3_bottom + 1]
            y_bottom = planes_params[idx3_bottom + 2]
            im_bottom_cut_x = self.grads_x_top[row + 1][col] + x_bottom
            im_bottom_cut_y = self.grads_y_top[row + 1][col] + y_bottom
            diff_bottom += mse(im_0_x_bottom, im_bottom_cut_x) * im_bottom_cut_x.size
            diff_bottom += mse(im_0_y_bottom, im_bottom_cut_y) * im_bottom_cut_y.size
        if edges:
            lam = lam_edges
            if col == 0:
                diff_left += mse(im_0_x_left, 0) * im_0_x_left.size * lam
                diff_left += mse(im_0_y_left, 0) * im_0_y_left.size * lam
            if col == self.cols - 1:
                diff_right += mse(im_0_x_right, 0) * im_0_x_right.size * lam
                diff_right += mse(im_0_y_right, 0) * im_0_y_right.size * lam
            if row == 0:
                diff_top += mse(im_0_x_top, 0) * im_0_x_top.size * lam
                diff_top += mse(im_0_y_top, 0) * im_0_y_top.size * lam
            if row == self.rows - 1:
                diff_bottom += mse(im_0_x_bottom, 0) * im_0_x_bottom.size * lam
                diff_bottom += mse(im_0_y_bottom, 0) * im_0_y_bottom.size * lam
        diff_total = diff_left + diff_right + diff_top + diff_bottom
        if maximize is True:
            diff_total = diff_total * -1
        return diff_total

    def sides_tv_2(self, col, row, mean=True):
        # idx = col + row * self.cols
        # idx +- self.cols is bottom/top neighbor
        # idx +- 1 is right/left neighbor
        im_0_left = self.frames_left[row][col] + self.planes_left[row][col]
        im_0_right = self.frames_right[row][col] + self.planes_right[row][col]
        im_0_top = self.frames_top[row][col] + self.planes_top[row][col]
        im_0_bottom = self.frames_bottom[row][col] + self.planes_bottom[row][col]
        diff_left = diff_right = diff_top = diff_bottom = 0
        if mean is True:
            diff_left = self._comp_fun(im_0_left, im_0_left.mean())  # rmse(im_0_left, 0) * im_0_left.size * lam
            diff_right = self._comp_fun(im_0_right, im_0_right.mean())  # rmse(im_0_right, 0) * im_0_right.size * lam
            diff_top = self._comp_fun(im_0_top, im_0_top.mean())  # rmse(im_0_top, 0) * im_0_top.size * lam
            diff_bottom = self._comp_fun(
                im_0_bottom, im_0_bottom.mean()
            )  # rmse(im_0_bottom, 0) * im_0_bottom.size * lam
        else:
            diff_left = self._comp_fun(im_0_left, 0)  # rmse(im_0_left, 0) * im_0_left.size * lam
            diff_right = self._comp_fun(im_0_right, 0)  # rmse(im_0_right, 0) * im_0_right.size * lam
            diff_top = self._comp_fun(im_0_top, 0)  # rmse(im_0_top, 0) * im_0_top.size * lam
            diff_bottom = self._comp_fun(im_0_bottom, 0)  # rmse(im_0_bottom, 0) * im_0_bottom.size * lam
        diff_total = diff_left + diff_right + diff_top + diff_bottom
        return diff_total

    def tv_fitness(self, planes_params, maximize=False):
        im_full = self.arrange_grid_memory(planes_params)
        fitness = tv(im_full)
        return fitness

    def tv_fitness_2(self, planes_params, maximize=False):
        # self._calc_planes(planes_params)
        fitness = 0
        for row in range(self.rows):
            for col in range(self.cols):
                # fitness += self.plane_fitting_diffs(col, row, maximize, edges=True)
                fitness += self.sides_tv(col, row, planes_params, maximize)
        return fitness

    def minimize_tv(self, maxiter=100):
        y = np.linspace(-1, 1, self.height)
        x = np.linspace(-1, 1, self.width)
        self.x, self.y = np.meshgrid(x, y)
        # self.load_frames()
        self.load_grads()
        res = minimize(self.tv_fitness_2, self.planes_params, options={'maxiter': maxiter})
        self.planes_params = res.x
        return res

    def plane_fitting_grid(self, save=True):
        im_full = self.arrange_grid_memory(planes_params=self.planes_params)
        self.planes_params = np.zeros_like(self.planes_params)
        im_full, _ = plane_fitting(im_full)
        self.grid_to_frames(im_full)
        if save:
            self.frames_to_disk()
        self.description += '_gridPF'
        return im_full

    def grid_to_frames(self, im_full):
        # self.frames = [[0 for col in range(self.cols)] for row in range(self.rows)]
        # rows = np.split(im_full, self.rows, axis=0)
        # for row in range(self.rows):
        #     row_list = np.split(rows[row], self.cols, axis=1)
        #     for col in range(self.cols):
        #         self.frames[row][col] = row_list[col]
        self.frames = [0 for row in range(self.rows)]
        rows = np.split(im_full, self.rows, axis=0)
        for row_idx, row in enumerate(rows):
            row_list = np.split(row, self.cols, axis=1)
            self.frames[row_idx] = row_list

    def frames_to_disk(self):
        for row in range(self.rows):
            for col in range(self.cols):
                imsave(str(self.paths[row][col]), self.frames[row][col])

    def save_to_tiff(self):
        # paths_new = [['' for col in range(self.cols)] for row in range(self.rows)]
        paths_new = copy.deepcopy(self.paths)
        temp_dir = paths_new[0][0].parent / 'temp'
        if not temp_dir.exists():
            temp_dir.mkdir()
        for row in range(self.rows):
            for col in range(self.cols):
                paths_new[row][col] = temp_dir / 'phase_x{0:0>3}_y{1:0>3}.tiff'.format(col, row)
                im_0 = imread(str(self.paths[row][col]))[1, :, :]
                im_tiff = im_0 * self.sign
                imsave(str(paths_new[row][col]), im_tiff)
        self.paths = paths_new
        im = imread(str(self.paths[row][col]))
        self.height = im.shape[0]
        self.width = im.shape[1]
        print('Saved raw')

    def save_to_tiff_idx(self, with_preview=True):
        # paths_new = [['' for col in range(self.cols)] for row in range(self.rows)]
        paths_new = copy.deepcopy(self.paths)
        temp_dir = paths_new[0][0].parent / 'temp'
        if not temp_dir.exists():
            temp_dir.mkdir()
        for row in range(self.rows):
            for col in range(self.cols):
                paths_new[row][col] = temp_dir / 'phase_idx{0:0>3}.tiff'.format(row * self.cols + col)
                if with_preview:
                    im_0 = imread(str(self.paths[row][col]))[1, :, :]
                else:
                    im_0 = imread(str(self.paths[row][col]))
                im_tiff = im_0 * self.sign
                imsave(str(paths_new[row][col]), im_tiff)
        self.paths = paths_new
        im = imread(str(self.paths[row][col]))
        self.height = im.shape[0]
        self.width = im.shape[1]
        print('Saved raw')

    def save_to_tiff_idx_3d(self, temp_path=None, dx_avg=None, time_point=0, frames_mode='mean', overwrite=False):
        from utils_3d import extract_from_mat, convert_n_to_int16

        if 'crop_percentage' in self.params:
            crop_percentage = self.params['crop_percentage']
        else:
            crop_percentage = None
            fov_size = self.params['fov_size']
            fov_shift_yx = self.params['fov_shift_yx']
        paths_new = copy.deepcopy(self.paths)
        frames = copy.deepcopy(self.paths)
        frames_paths = copy.deepcopy(self.paths)
        if temp_path is None:
            temp_path = paths_new[0][0].parent.parent / 'temp'
        if not temp_path.exists():
            temp_path.mkdir()
        tile_config_path = temp_path / 'tileConfig.txt'
        if tile_config_path.exists():
            tile_config_path.unlink()
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                print(f'save_to_tiff_idx_3d: {i} / {self.rows * self.cols}')
                paths_new[row][col] = temp_path / f'REC_idx{i:03d}.tiff'
                frames_path = f'frames_idx{i:03d}_{frames_mode}.tiff'
                frames_path = temp_path / frames_path
                frames_paths[row][col] = frames_path
                if overwrite or (not paths_new[row][col].exists()):  # (paths_new[row][col].suffix not in '.tiff') or
                    if paths_new[row][col].suffix not in '.tiff':
                        vol_16, _ = extract_from_mat(str(self.paths[row][col]), load_rec=True)
                        vol_16 = convert_n_to_int16(vol_16)
                        if crop_percentage is not None:
                            margin = int(round(crop_percentage / 2 * vol_16.shape[0]))
                            vol_16 = vol_16[margin:-margin, margin:-margin, margin:-margin]
                        else:
                            vol_16 = sub_mat_fov(vol_16, fov_size, shift_yx=fov_shift_yx)
                        if frames_mode == 'max':
                            frames[row][col] = vol_16.max(axis=0)
                        elif frames_mode == 'center':
                            frames[row][col] = vol_16[vol_16.shape[2] // 2]
                        else:
                            frames[row][col] = vol_16.mean(axis=0)
                        imsave(str(paths_new[row][col]), vol_16)
                        imsave(str(frames_path), frames[row][col])
                        self.height = vol_16.shape[1]
                        self.width = vol_16.shape[2]
        self.paths = paths_new
        self.frames = frames
        self.frames_paths = frames_paths
        print('Saved raw')

    def extract_tile_config(self, temp_path=None, dx_avg=None, idx_min=0, time_point=0):
        from utils_3d import extract_from_mat

        if temp_path is None:
            temp_path = self.paths[0][0].parent.parent / 'temp'
        if not temp_path.exists():
            temp_path.mkdir()
        tile_config_path = temp_path / 'tileConfig.txt'
        if tile_config_path.exists():
            tile_config_path.unlink()
        with open(tile_config_path, 'w+') as f:
            f.write('dim=3')
            f.write('\n')
            for row in range(self.rows):
                for col in range(self.cols):
                    i = row * self.cols + col
                    print(f'extract_tile_config: {i} / {self.rows * self.cols}')
                    _, params = extract_from_mat(str(self.paths[row][col]), load_rec=False)
                    if dx_avg is None:
                        pos_x = params['pos_x'] / params['dx']
                        pos_y = params['pos_y'] / params['dx']
                        pos_z = params['pos_z'] / params['dx']
                    else:
                        pos_x = params['pos_x'] / dx_avg
                        pos_y = params['pos_y'] / dx_avg
                        pos_z = params['pos_z'] / dx_avg
                    f.write('{};{};({:.5f}, {:.5f}, {:.5f})'.format(i - idx_min, time_point, pos_x, pos_y, pos_z))
                    f.write('\n')
        with open(temp_path / 'info.txt', 'w+') as f:
            if dx_avg is None:
                f.write('dx = {:.5f} um\n'.format(params['dx']))
            else:
                f.write('dx = {:.5f} um\n'.format(dx_avg))
            f.write('n = {:.5f}\n'.format(params['n_imm']))

    def rebuild_frames(self, frames_mode='mean'):
        from utils_3d import extract_from_mat, convert_n_to_int16

        for row in range(self.rows):
            for col in range(self.cols):
                idx = col + row * self.cols
                print(f'rebuild_frames: {idx} / {self.rows * self.cols}')
                if type(frames_mode) is int:
                    self.frames[row][col] = imread(str(self.paths[row][col]), key=frames_mode).astype(np.float64)
                else:
                    vol_16 = imread(str(self.paths[row][col])).astype(np.float64)
                    if frames_mode == 'max':
                        self.frames[row][col] = vol_16.max(axis=0).copy()
                    elif frames_mode == 'center':
                        self.frames[row][col] = vol_16[vol_16.shape[2] // 2].copy()
                    else:
                        self.frames[row][col] = vol_16.mean(axis=0).copy()
        self.load_frames_sides_from_frames()

    def save_frames_to_disk(self, suffix=None):
        for row in range(self.rows):
            for col in range(self.cols):
                # frame_path = self.frames_paths[row][col]
                # if suffix is not None:
                #     frame_path = frame_path.parent / (frame_path.stem + '_' + suffix + frame_path.suffix)
                i = row * self.cols + col
                print(f'save_frames_to_disk: {i} / {self.rows * self.cols}')
                frame_path = f'frames_idx{i:03d}'
                if suffix is not None:
                    frame_path += f'_{suffix}'
                frame_path += '.tiff'
                frame_path = self.frames_paths[row][col].parent / frame_path
                imsave(frame_path, self.frames[row][col])

    def load_frames_from_disk(self, suffix=None):
        for row in range(self.rows):
            for col in range(self.cols):
                # frame_path = self.frames_paths[row][col]
                # if suffix is not None:
                #     frame_path = frame_path.parent / (frame_path.stem + '_' + suffix + frame_path.suffix)
                i = row * self.cols + col
                print(f'load_frames_from_disk: {i} / {self.rows * self.cols}')
                frame_path = f'frames_idx{i:03d}'
                if suffix is not None:
                    frame_path += f'_{suffix}'
                frame_path += '.tiff'
                frame_path = self.frames_paths[row][col].parent / frame_path
                self.frames[row][col] = imread(frame_path)

    def rename_frames_paths(self, frames_mode=None, rename_on_disk=False):
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                print(f'rename_frames_paths: {i} / {self.rows * self.cols}')
                frame_path = f'frames_idx{i:03d}'
                if frames_mode is not None:
                    frame_path += f'_{frames_mode}'
                frame_path += '.tiff'
                frame_path = self.frames_paths[row][col].parent / frame_path
                if rename_on_disk:
                    self.frames_paths[row][col].rename(frame_path)
                self.frames_paths[row][col] = frame_path

    def rename_paths(self, rec_mode=None, rename_on_disk=False):
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                print(f'rename_paths: {i} / {self.rows * self.cols}')
                rec_path = f'REC_idx{i:03d}'
                if rec_mode is not None:
                    rec_path += f'_{rec_mode}'
                rec_path += '.tiff'
                rec_path = self.paths[row][col].parent / rec_path
                if rename_on_disk:
                    self.paths[row][col].rename(rec_path)
                self.paths[row][col] = rec_path

    # def load_frames(self):
    # self.frames = [[imread(self.frames_paths[row][col]) for col in range(self.cols)] for row in range(self.rows)]

    def resave_as_type(self, type=np.float64):
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im = np.asarray(im, dtype=type)
                imsave(str(self.paths[row][col]), im)
        print('Resaved with different data type')

    def resave_to_tiff_with_offsets_idx(self, offsets=None, keep_copy=True, suffix='offsets_1'):
        if keep_copy:
            self.paths_old = copy.deepcopy(self.paths)
        if offsets is None:
            offsets = self.offsets
        self.offsets_backup = offsets.copy()
        im = imread(str(self.paths[0][0]))
        if im.dtype == np.uint16:
            im_uint16 = True
            offsets = np.round(copy.deepcopy(offsets))
            # offsets_sign = np.sign(offsets)
            offsets = np.asarray(offsets, dtype=np.int32)
        else:
            im_uint16 = False
        # name_suffix = f'_offsets_{num}'
        # pattern = self.params['name_pattern']
        # pattern_new = pattern.split('.tiff')[0] + name_suffix + '.tiff'
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                print(f'resave_to_tiff_with_offsets_idx: {i} / {self.rows * self.cols}')
                im = imread(str(self.paths[row][col]))
                if im_uint16:
                    im = im.astype(np.int32)
                    # if offsets_sign[idx] > 0:
                    #     im_corr = im + offsets[idx]
                    # else:
                    #     im_corr = im - offsets[idx]
                    # if im_corr.dtype != np.uint16:
                    #     test = 1
                # else:
                idx = col + row * self.cols
                im_corr = im + offsets[idx]
                if im_uint16:
                    im_corr = np.round(im_corr).astype(np.uint16)
                if keep_copy:
                    rec_path = f'REC_idx{i:03d}'
                    # if suffix is not None:
                    rec_path += f'_{suffix}'
                    rec_path += '.tiff'
                    rec_path = self.paths[row][col].parent / rec_path
                    # path_curr = Path(self.paths[row][col])
                    # path_new = str(path_curr.parent / path_curr.stem) + name_suffix + path_curr.suffix
                    self.paths[row][col] = rec_path
                    imsave(str(rec_path), im_corr)
                else:
                    imsave(str(self.paths[row][col]), im_corr)
                test = 1
        self.offsets = np.zeros_like(self.offsets)
        self.description += '_optOffsets'
        print('Saved with offsets')
        # return pattern_new

    def resave_to_tiff_with_planes_idx(self, planes_params=None, keep_copy=True, suffix='planes_1'):
        if keep_copy:
            self.paths_old = copy.deepcopy(self.paths)
        if planes_params is None:
            planes_params = self.planes_params
        self.planes_params_backup = planes_params.copy()
        im = imread(str(self.paths[0][0]))
        if im.dtype == np.uint16:
            im_uint16 = True
            planes_params = np.round(copy.deepcopy(planes_params))
            # offsets_sign = np.sign(offsets)
            planes_params = np.asarray(planes_params, dtype=np.int32)
        else:
            im_uint16 = False
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                print(f'resave_to_tiff_with_planes_idx: {i} / {self.rows * self.cols}')
                im = imread(str(self.paths[row][col]))
                idx3 = (col + row * self.cols) * 3
                plane_params = (planes_params[idx3], planes_params[idx3 + 1], planes_params[idx3 + 2])
                plane = self._calc_plane(plane_params, override_downsample=True)
                if im_uint16:
                    im = im.astype(np.int32)
                    plane = np.round(plane).astype(np.int32)
                idx = col + row * self.cols
                im_corr = im + plane
                if im_uint16:
                    im_corr = np.round(im_corr).astype(np.uint16)
                if keep_copy:
                    rec_path = f'REC_idx{i:03d}'
                    # if suffix is not None:
                    rec_path += f'_{suffix}'
                    rec_path += '.tiff'
                    rec_path = self.paths[row][col].parent / rec_path
                    # path_curr = Path(self.paths[row][col])
                    # path_new = str(path_curr.parent / path_curr.stem) + name_suffix + path_curr.suffix
                    self.paths[row][col] = rec_path
                    imsave(str(rec_path), im_corr)
                else:
                    imsave(str(self.paths[row][col]), im_corr)
                test = 1
        planes_params = np.zeros_like(planes_params)
        self.iter_count = 0
        self.description += '_optPF'
        print('Saved with planes')

    def _compare_tiffs(self, prefix='REC_idx', suffix=None):
        offsets = np.zeros(self.rows * self.cols)
        planes_params = np.zeros(self.rows * self.cols * 3)
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                idx3 = (col + row * self.cols) * 3
                tiff_path = f'{prefix}{i:03d}'
                if suffix is not None:
                    tiff_path += f'_{suffix}'
                tiff_path += '.tiff'
                tiff_path = self.paths[row][col].parent / tiff_path
                im_ref = imread(str(self.paths[row][col]))
                im = imread(str(tiff_path))
                im_diff = im_ref - im
                im_diff = im_diff.astype(np.float64)
                offsets[i] = im_diff.mean()
                planes_params[idx3] = offsets[i]
                _, slopes = gradient_PF_lf(im_diff[im_diff.shape[0] // 2])
                planes_params[idx3 + 1] = -slopes[0]
                planes_params[idx3 + 2] = -slopes[1]
        return offsets, planes_params

    def delete_tiffs(self):
        for row in range(self.rows):
            for col in range(self.cols):
                try:
                    os.remove(self.paths[row][col])
                except OSError:
                    pass
        print('Files cleaned up')

    def remove_legendre_ls(self, deg=2, unwrap_err_filt=False, seg=False, new_seg=False):
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                if unwrap_err_filt or seg == True:
                    im_nan = copy.deepcopy(im)
                    if new_seg == False:
                        if unwrap_err_filt == True:
                            error_mask, _ = unwrapping_error_segmentation(im)
                            im_nan[error_mask == 1] = np.nan
                        if seg == True:
                            cell_mask = mog_cells(im_nan)
                            im_nan[cell_mask == 1] = np.nan
                    else:
                        mask = mog_and_thr_segmentation(im)
                        im_nan[mask == 1] = np.nan
                    aberr, _ = legendre_fitting_nan(im_nan, deg)
                else:
                    aberr, _ = legendre_fitting_nan(im, deg)
                im = im - aberr
                imsave(str(self.paths[row][col]), im)
        # self.description += '_legendre'
        print('Saved after Legendre')

    def remove_legendre_ls_2(self, deg=2, unwrap_err_filt=None, segment_cells=None, verbose=False):
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im_nan = copy.deepcopy(im)
                mean_bg = 0
                if unwrap_err_filt is not None:
                    error_mask = unwrap_err_filt(im)
                    im_nan[error_mask == 1] = np.nan
                if segment_cells is not None:
                    cell_mask = segment_cells(im=im)
                    mean_bg = np.ma.array(im, mask=cell_mask).mean()
                    cell_mask_sum = cell_mask.sum()
                    # if cell_mask_sum -
                    if cell_mask_sum > im.size * 0.7:
                        # if there is more than 70% of cells in FoV, use only cell regions
                        im_nan[cell_mask == 0] = np.nan
                        if verbose is True:
                            print('Fit on cells')
                    else:
                        # otherwise use background
                        im_nan[cell_mask == 1] = np.nan
                        if verbose is True:
                            print('Fit on background')
                aberr, _ = legendre_fitting_nan(im_nan, deg)
                im = im - aberr - mean_bg
                imsave(str(self.paths[row][col]), im)
        if deg > 1:
            self.description += '_legendre'
        else:
            self.description += '_PF'
        print('Saved after Legendre')

    def remove_legendre_average(self, deg=2, unwrap_err_filt=None, segment_cells=None):
        iteration = 1
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im_nan = copy.deepcopy(im)
                if unwrap_err_filt is not None:
                    error_mask = unwrap_err_filt(im)
                    im_nan[error_mask == 1] = np.nan
                if segment_cells is not None:
                    cell_mask = segment_cells(im=im)
                    cell_mask_sum = cell_mask.sum()
                    # if cell_mask_sum -
                    if cell_mask_sum > im.size * 0.7:
                        # if there is more than 70% of cells in FoV, use only cell regions
                        im_nan[cell_mask == 0] = np.nan
                    else:
                        # otherwise use background
                        im_nan[cell_mask == 1] = np.nan
                aberr, _ = legendre_fitting_nan(im_nan, deg)
                self.mean_aberr, iteration = recursive_mean(aberr, self.mean_aberr, iteration)
                # print('Averaged x - {} y - {}'.format(col, row))
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im -= self.mean_aberr
                imsave(str(self.paths[row][col]), im)
        # self.description += '_legendre'
        print('Saved after Legendre')

    def remove_slope(self, unwrap_err_filt=None, segment_cells=None, verbose=False):
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                error_mask = cell_mask = np.zeros_like(im)
                if unwrap_err_filt is not None:
                    error_mask, _ = unwrap_err_filt(im)
                if segment_cells is not None:
                    cell_mask = segment_cells(im=im)
                    cell_mask_sum = cell_mask.sum()
                    # if cell_mask_sum -
                    if cell_mask_sum > im.size * 0.9:
                        # if there is more than 90% of cells in FoV, use only cell regions
                        cell_mask = ~cell_mask
                        if verbose is True:
                            print('Fit on cells')
                    else:
                        # otherwise use background
                        if verbose is True:
                            print('Fit on background')
                mask_sum = error_mask + cell_mask
                aberr = slope_fitting(im, mask_sum)
                im = im - aberr
                imsave(str(self.paths[row][col]), im)
        # self.description += '_legendre'
        print('Saved after slope fitting')

    def gradient_slopes(self, sigma=20):
        from scipy.ndimage import gaussian_filter, median_filter

        def filter_gradient(grad):
            grad_filt = gaussian_filter(grad, sigma=sigma)
            # grad_filt = median_filter(grad, size=sigma)
            # thr = threshold_otsu(grad_filt)
            # _, grad_filt = plane_fitting(grad_filt)
            slope = np.median(grad_filt)
            return slope

        y = np.arange(self.height) - self.height // 2
        x = np.arange(self.width) - self.width // 2
        x, y = np.meshgrid(x, y)
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                grad_y, grad_x = np.gradient(im)
                y_slope = filter_gradient(grad_y)
                im -= y_slope * y
                x_slope = filter_gradient(grad_x)
                im -= x_slope * x
                imsave(str(self.paths[row][col]), im)
        self.description += '_gradPF'
        print('Saved after Gaussian filtered slope estimation')

    def median_slope(self, size=20):
        from scipy.ndimage import median_filter

        y = np.arange(self.height) - self.height // 2
        x = np.arange(self.width) - self.width // 2
        x, y = np.meshgrid(x, y)
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                grad_y, grad_x = np.gradient(im)
                grad_y_med = median_filter(grad_y, size=size)
                y_slope = np.median(grad_y_med)
                im -= y_slope * y
                grad_x_med = median_filter(grad_x, size=size)
                x_slope = np.median(grad_x_med)
                im -= x_slope * x
                imsave(str(self.paths[row][col]), im)
        print('Saved after slope centering')

    def gaussian_filter(self, sigma=20):
        from scipy.ndimage import gaussian_filter

        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im_filt = gaussian_filter(im, sigma=sigma, mode='nearest')
                imsave(str(self.paths[row][col]), im - im_filt)
        print('Saved after slope centering')

    def remove_svd(self, unwrap_err_filt=False, seg=False):
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                if unwrap_err_filt or seg == True:
                    im_nan = copy.deepcopy(im)
                    if unwrap_err_filt == True:
                        error_mask, _ = unwrapping_error_segmentation(im)
                        im_nan[error_mask == 1] = np.inf
                    if seg == True:
                        cell_mask = mog_cells(im_nan)
                        im_nan[cell_mask == 1] = np.inf
                    im_ph = np.exp(1j * im_nan)
                    im = svd_correction(im_ph)
                else:
                    im_ph = np.exp(1j * im)
                    im = svd_correction(im_ph)
                imsave(str(self.paths[row][col]), im)
        self.description += '_PCA'
        print('Saved after PCA')

    def segment_cells_ws(self, im):
        background_g = 1
        cell_g = 4
        orig_min_max = self.find_min_max()
        im_rescaled = rescale_intensity(im, in_range=orig_min_max, out_range=(0, 1))
        im_rescaled[im_rescaled < 0] = 0
        im_rescaled[im_rescaled > 1] = 1
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

    def fit_plane(self, unwrap_err_filt=False, seg=False, thr=None, show=False):
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                if unwrap_err_filt or seg == True:
                    im_nan = copy.deepcopy(im)
                    if unwrap_err_filt == True:
                        error_mask, _ = unwrapping_error_segmentation(im)
                        im_nan[error_mask == 1] = np.nan
                    if seg == True:
                        cell_mask = mog_cells(im_nan)
                        if thr is not None:
                            prc_of_masked_cells = np.sum(cell_mask) / (cell_mask.shape[0] * cell_mask.shape[1])
                            if prc_of_masked_cells > thr:
                                cell_mask == np.zeros_like(cell_mask)
                        im_nan[cell_mask == 1] = np.nan
                    _, plane = plane_fitting(im_nan)
                else:
                    _, plane = plane_fitting(im_nan)
                im = im - plane
                if show == True:
                    subplot_row([im + plane, im, cell_mask], show=True)
                imsave(str(self.paths[row][col]), im)
        if seg == True:
            self.description += '_segPF'
        else:
            self.description += '_PF'
        print('Saved after PF')

    def remove_precalc_average_aberrations(self, abr):
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                print(f'remove_precalc_average_aberrations: {i} / {self.rows * self.cols}')
                im = imread(str(self.paths[row][col]))
                im_removed = im - abr
                imsave(str(self.paths[row][col]), im_removed)
        # self.description += '_AME'

    def return_average_aberrations(self):
        im = imread(str(self.paths[0][0]))
        abr = np.zeros_like(im)
        iteration = 1
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                abr, iteration = recursive_mean(im, abr, iteration)
        return abr

    def return_average_aberrations_from_frames(self, plane_fitting=False):
        im = self.frames[0][0]
        abr = np.zeros_like(im)
        iteration = 1
        for row in range(self.rows):
            for col in range(self.cols):
                im = self.frames[row][col]
                abr, iteration = recursive_mean(im, abr, iteration)
        if plane_fitting:
            abr = gradient_PF(abr)
        abr -= abr.mean()
        return abr

    def remove_precalc_aberrations_from_frames(self, abr):
        for row in range(self.rows):
            for col in range(self.cols):
                self.frames[row][col] -= abr

    def plane_fit_frames(self):
        for row in range(self.rows):
            for col in range(self.cols):
                i = row * self.cols + col
                idx3 = (col + row * self.cols) * 3
                print(f'plane_fit_frames: {i} / {self.rows * self.cols}')
                _, slopes = gradient_PF_lf(self.frames[row][col])
                self.planes_params[idx3 + 1] = slopes[0]
                self.planes_params[idx3 + 2] = slopes[1]

    def average_aberrations(self, remove_unwrap_errors=True, show=False):
        im = imread(str(self.paths[0][0]))
        abr = np.zeros_like(im)
        iteration = 1
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                if remove_unwrap_errors == True:
                    error_mask, im2 = unwrapping_error_segmentation(im)
                    if show == True:
                        if np.sum(error_mask) > 0:
                            subplot_row([im, im2])
                    im = im2
                # abr = abr + im
                abr, iteration = recursive_mean(im, abr, iteration)
        # abr = abr / (self.rows * self.cols)
        grid_size = self.height * self.width
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im_cleared = im - abr
                im_cleared = im_cleared * (
                    grid_size / (grid_size - 1)
                )  # correction due to removal of the frame itself contained in average image
                imsave(str(self.paths[row][col]), im_cleared)
        path = os.path.dirname(self.paths[row][col])
        path = Path(path) / 'averaged_aberrations.tiff'
        abr_tiff = []
        abr_tiff.append(abr)
        abr_tiff = np.asarray(abr_tiff, dtype=np.float32)
        imsave(str(path), abr_tiff)
        self.description += '_AMEM'
        print('Saved after AMEM')

    def double_exposure(self):
        abr = imread(str(self.abr_path))[1]
        for row in range(self.rows):
            for col in range(self.cols):
                im = imread(str(self.paths[row][col]))
                im_cleared = im - abr
                imsave(str(self.paths[row][col]), im_cleared)
        self.description += '_DE'
        print('Saved after DE')


def average_phase_timelapse(paths_timelapses, change_sign=1, with_preview=True):
    no_timelapses = len(paths_timelapses)
    # no_rows = len(paths_timelapse[0])
    # no_cols = len(paths_timelapse[0][0])
    if with_preview:
        avg_abr = np.zeros_like(imread(paths_timelapses[0][0][0])[1])
    else:
        avg_abr = np.zeros_like(imread(paths_timelapses[0][0][0]))
    iteration = 1
    for i, paths_timelapse in enumerate(paths_timelapses):
        print(f'{i} out of {no_timelapses} timepoints')
        for paths_row in paths_timelapse:
            for path in paths_row:
                if with_preview:
                    im = imread(str(path))[1, :, :]
                else:
                    im = imread(str(path))
                avg_abr, iteration = recursive_mean(im * change_sign, avg_abr, iteration)
    return avg_abr


def gradient_PF_lf(im, sigma=10):
    from scipy.ndimage import gaussian_filter, median_filter

    def filter_gradient(grad):
        grad_filt = gaussian_filter(grad, sigma=sigma)
        # grad_filt = median_filter(grad, size=sigma)
        # thr = threshold_otsu(grad_filt)
        slope = np.median(grad_filt)
        return slope

    (height, width) = im.shape
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    x, y = np.meshgrid(x, y)
    dy = 2 / height
    dx = 2 / width
    grad_y, grad_x = np.gradient(im, dy, dx)
    y_slope = filter_gradient(grad_y)
    im -= y_slope * y
    x_slope = filter_gradient(grad_x)
    im -= x_slope * x
    return im, (-x_slope, -y_slope)


def correlate_images(im0, im1, upsample_factor=1):
    shifts, _, _ = register_translation(im0, im1, upsample_factor=upsample_factor)
    y = int(shifts[0])
    x = int(shifts[1])
    height, width = im0.shape
    if y > 0:
        im0_cut = im0[y:, :]
        im1_cut = im1[: height - y, :]
    else:
        y = -y
        im0_cut = im0[: height - y, :]
        im1_cut = im1[y:, :]
    if x > 0:
        im0_cut = im0_cut[:, x:]
        im1_cut = im1_cut[:, : width - x]
    else:
        x = -x
        im0_cut = im0_cut[:, : width - x]
        im1_cut = im1_cut[:, x:]
    return im0_cut, im1_cut


def threshold_2(ims, thr_fun=np.nanpercentile, percentile=40, stat=np.mean, im_thr=None):
    vals = []
    for im in ims:
        im2 = np.copy(im)
        if im_thr is not None:
            mask = im_thr(im)
            im2[mask == 1] = np.nan
        if thr_fun == np.nanpercentile:
            bot_val = thr_fun(im2, percentile)
            top_val = thr_fun(im2, 100 - percentile)
            im2 = im2.flatten()
            im2 = im2[im2 < bot_val]
            # im2 = im2[im2 > bot_val]
            # im2 = im2[im2 < top_val]
        else:
            # im2 = im2.flatten()
            # im2 = im2[np.isfinite(im2)]
            thr_val = thr_fun(im2)
            im2 = im2[im2 > thr_val]
            # im2 = im2[im2 < thr_val]
        val = stat(im2)
        vals.append(val)
    return vals


def mog_cells(img):
    im = copy.deepcopy(img)
    im2 = np.reshape(im, -1)
    im2 = im2[np.isfinite(im2)]
    median_val = np.median(im2)
    im[~np.isfinite(im)] = median_val
    scale = 6
    top_val = np.max(im)
    bot_val = np.min(im)
    range_val = top_val - bot_val
    im_res = rescale(set_range(im, 0, 1), 1 / scale)
    im_mog = magnitude_of_gradient(im_res) * range_val
    # im_mog = magnitude_of_gradient(rescale(set_range(im, 0, 1), 1 / 6, anti_aliasing=True))
    im_b = np.zeros_like(im_mog)
    im_mog_temp = np.reshape(im_mog, -1)
    im_mog_temp = im_mog_temp[im_mog_temp != np.nan]
    thr_val = filters.threshold_li(im_mog_temp)  # * 1.2 * range_val
    # filters.try_all_threshold(im_mog)
    im_b[im_mog > thr_val] = 1
    # im_b = ndimage.binary_opening(im_b, structure=np.ones((1, 1)))
    im_b = filters.median(im_b, selem=np.ones((2, 2)))
    im_b = filters.median(im_b, selem=np.ones((3, 3)))
    # # im_b = ndimage.binary_dilation(im_b, structure=np.ones((3, 3)))
    im_b = ndimage.binary_closing(im_b, structure=np.ones((3, 3)))
    im_b = ndimage.binary_fill_holes(im_b)
    im_b = ndimage.binary_dilation(im_b, structure=np.ones((3, 3)))
    im_b = resize(im_b, im.shape)
    im_b[im_b > 0.5] = 1
    im_b[im_b <= 0.5] = 0
    # im_b = filters.median(im_b, selem=np.ones((3, 3)))
    return im_b


def mog_and_thr_segmentation(img):
    im = np.copy(img)
    final_mask = np.zeros_like(im)
    unwrapping_mask, im = unwrapping_error_segmentation(im)
    im[unwrapping_mask == 1] = np.nan
    background_mask = mog_background(im)
    background_mask[unwrapping_mask == 1] = 0
    # im = thr_cells(im)
    background_fill = np.sum(background_mask) / background_mask.size
    # background_fills.append(background_fill)
    # if background_fill < 0.2:
    if background_fill < 0.4:
        im[background_mask == 1] = np.nan
        cells_mask = thr_cells(im)
        final_mask = cells_mask
        # method.append('cells')
    else:
        # cells_mask = np.zeros_like(im)
        final_mask[background_mask == 0] = 1
        # method.append('background')
    return final_mask


def mog_background(img):
    im = copy.deepcopy(img)
    im2 = np.reshape(im, -1)
    im2 = im2[np.isfinite(im2)]
    median_val = np.median(im2)
    im[~np.isfinite(im)] = median_val
    scale = 6
    top_val = np.max(im)
    bot_val = np.min(im)
    range_val = top_val - bot_val
    im_res, im_min, im_max = set_range(im, 0, 1)
    im_res = rescale(im_res, 1 / scale)
    im_mog = magnitude_of_gradient(im_res) * range_val
    # im_mog = magnitude_of_gradient(rescale(set_range(im, 0, 1), 1 / 6, anti_aliasing=True))
    im_b = np.zeros_like(im_mog)
    im_mog_temp = np.reshape(im_mog, -1)
    im_mog_temp = im_mog_temp[im_mog_temp != np.nan]
    # thr_val = filters.threshold_li(im_mog_temp)  # * 1.2 * range_val
    thr_val = 0.11
    # filters.try_all_threshold(im_mog)
    im_b[im_mog < thr_val] = 1
    im_b = ndimage.binary_opening(im_b, structure=np.ones((5, 5)))
    # im_b = filters.median(im_b, selem=np.ones((2, 2)))
    im_b = resize(im_b, im.shape)
    im_b[im_b > 0.5] = 1
    im_b[im_b <= 0.5] = 0
    return im_b


def thr_cells(img):
    im = np.copy(img)
    im = np.ndarray.flatten(im)
    im = im[np.isfinite(im)]
    # thr_val = filters.threshold_li(im)
    thr_val = np.percentile(im, 50)
    bot_val = np.percentile(im, 2)
    mask = np.zeros_like(img)
    mask[img > thr_val] = 1
    mask[img < bot_val] = 1
    mask[~np.isfinite(img)] = 1
    return mask


def recursive_mean(additional_data, mean_so_far, iteration):
    if iteration == 1:
        mean_so_far = additional_data
    else:
        mean_so_far = (iteration - 1) / iteration * mean_so_far + additional_data / iteration
    iteration += 1
    return mean_so_far, iteration

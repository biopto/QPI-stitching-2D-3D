__author__ = 'Piotr Stępień'

from aberrations import legendre_fitting, legendre_fitting_nan
from stitching import mog_cells, unwrapping_error_segmentation, mog_and_thr_segmentation
import numpy as np
from denoise import recursive_mean
from skimage.io import imread, imsave
from image_processing import set_range, im_preview
from visualization import imshow_timelapse, imshow_paper
import os
import shutil
import copy


class TimelapseCollection:
    def __init__(self, home_path, name_base, ref_num, file_format, start_num, end_num, sign, ph_px_size):
        self.home_path = home_path
        self.name_base = name_base
        self.ref_num = ref_num
        self.file_format = file_format
        self.start_num = start_num
        self.end_num = end_num
        self.sign = sign
        self.ph_px_size = ph_px_size
        self.paths = []
        for i in range(self.end_num - self.start_num + 1):
            self.paths.append(
                os.path.join(home_path, name_base + ref_num + '_' + str(self.start_num + i).zfill(3) + self.file_format)
            )
        self.output_folder = os.path.join(home_path, 'timelapse')
        self.temp_folder = os.path.join(self.output_folder, 'temp')
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.isdir(self.temp_folder):
            os.makedirs(self.temp_folder)

    def average_full(self, filtration_method='legendre', order=20, modes=2):
        print('Averaging')
        im_avg = np.zeros_like(imread(self.paths[0])[1])
        for i, p in enumerate(self.paths):
            im = imread(p)[1] * self.sign
            im_avg, _ = recursive_mean(im, im_avg, i + 1)
        lf_res = 0
        if filtration_method == 'legendre':
            print('Legendre filtration, order = ' + str(order))
            lf_res, _ = legendre_fitting(im_avg, order)
        else:
            raise ValueError('ERROR: Accepted filtration methods are "legendre" and ...')
        self.im_noise = im_avg - lf_res
        print('Saving filtered files')
        for i, p in enumerate(self.paths):
            im_ph = imread(p)[1]
            im_ph = im_ph - self.im_noise
            p = os.path.join(self.temp_folder, 'phase_' + str(i) + '.tiff')
            imsave(p, im_ph)
            self.paths[i] = p
        im_file = im_preview(self.im_noise)
        imsave(os.path.join(self.output_folder, 'noise.tiff'), im_file)
        test = 1

    def average_batch(self):
        pass

    def show_differences(self):
        for i in range(len(self.paths) - 1):
            im_0 = imread(self.paths[i])
            im_1 = imread(self.paths[i + 1])
            plt.figure(), plt.imshow(im_1 - im_0), plt.colorbar(), plt.show()

    def remove_legendre_ls(self, deg=2, unwrap_err_filt=False, seg=False, new_seg=False):
        print('Individual Legendre filtration, order = ' + str(deg))
        for p in self.paths:
            im = imread(p)
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
            imsave(p, im)
        # self.description += '_legendre'
        print('Saved after Legendre')

    def create_frames(self, colormap='cividis', colorbar='True'):
        print('Saving created frames')
        for p in self.paths:
            im = imread(p)
            dot_pos = p.rfind('.')
            p_png = p[:dot_pos] + '.png'
            imshow_timelapse(
                im, outfile=p_png, px_size=self.ph_px_size, colormap=colormap, colorbar=colorbar, show=False
            )
            os.remove(p)

    def delete_temp(self):
        print('Files clean up')
        for p in self.paths:
            try:
                shutil.rmtree(self.temp_folder)
                # if os.path.dirname(p) == self.temp_folder:
                #     os.remove(p)
            except OSError:
                pass
        print('Files cleaned up')


def calc_px_size(
    sample_im_path, tukey_total_size=0.1, magnification=50.9, cam_px_size=3.45, cam_height=2056, cam_width=2464
):
    im_sample = imread(sample_im_path)[1]
    ph_height, ph_width = im_sample.shape
    shrink_factor = 1 - tukey_total_size
    height = round(cam_height * shrink_factor)
    width = round(cam_width * shrink_factor)
    real_height = height * cam_px_size / magnification
    real_width = width * cam_px_size / magnification
    ph_px_height = real_height / ph_height
    ph_px_width = real_width / ph_width
    ph_px_size = [ph_px_height, ph_px_width]
    return ph_px_size


def timelapse_2018_11_21_15_47():
    home_path = 'D:\\Python\\Python3.6\\Timelapse\\2018.11.21 HaCaT\\timelapse-2018-11-21#15_47\\'
    name_base = 'phase_ref'
    ref_num = '245'
    file_format = '.tiff'
    start_num = 2
    end_num = 1021
    sign = 1

    tukey_total_size = 0.1
    magnification = 50.9
    cam_px_size = 3.45
    cam_height = 2056
    cam_width = 2464
    sample_im_path = os.path.join(home_path, name_base + ref_num + '_' + str(2).zfill(3) + file_format)
    ph_px_size = calc_px_size(sample_im_path, tukey_total_size, magnification, cam_px_size, cam_height, cam_width)
    params = [home_path, name_base, ref_num, file_format, start_num, end_num, sign, ph_px_size]
    return params


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['image.cmap'] = 'cividis'
    params = timelapse_2018_11_21_15_47()
    images = TimelapseCollection(*params)
    imshow_timelapse(imread(images.paths[0])[1], px_size=params[-1], colormap='cividis', colorbar=True, show=False)
    # imshow_paper(imread(images.paths[0])[1])
    images.average_full()
    # images.show_differences()
    # images.remove_legendre_ls(deg=2, unwrap_err_filt=True, seg=True, new_seg=True)
    images.create_frames(colormap='cividis', colorbar=True)
    images.delete_temp()
    test = 1

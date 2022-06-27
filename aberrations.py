__author__ = 'Piotr Stępień'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.filters.thresholding import threshold_li, threshold_otsu
from skimage.filters import sobel
from skimage.io import imread, imsave
from skimage.restoration import unwrap_phase
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, mode
from image_processing import magnitude_of_gradient, batch_round, ims_range
import os
import itertools
import matplotlib

### Usefull links:
#  https://stackoverflow.com/questions/33193841/image-reconstruction-based-on-zernike-moments-using-mahotas-and-opencv#33339289
#  http://mahotas.readthedocs.io
#  https://github.com/Sterncat/opticspy
#  https://github.com/tvwerkhoven/libtim-py


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def legendre_2d_2(x, y, deg_x, deg_y):
    c_x = np.zeros(deg_x + 1)
    c_y = np.zeros(deg_y + 1)
    c_x[-1] = 1
    c_y[-1] = 1
    leg_x = np.polynomial.legendre.legval(x, c_x)
    leg_y = np.polynomial.legendre.legval(y, c_y)
    return np.matmul(leg_y.reshape(-1, 1), leg_x.reshape(1, -1))


def legendre_2d(x, y, deg_x, deg_y):
    c = np.zeros([deg_y + 1, deg_x + 1])
    c[-1, -1] = 1
    leg = np.polynomial.legendre.leggrid2d(x, y, c)  # for some reason does not work as legval2d used to
    return leg


def legendre_fitting(Y, max_degree):
    m = Y.shape[0]
    n = Y.shape[1]

    # y, x = np.mgrid[np.linspace(-1, 1, m), np.linspace(-1, 1, n)]
    y = np.linspace(-1, 1, m)
    x = np.linspace(-1, 1, n)
    c = np.zeros([max_degree + 1, max_degree + 1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if i + j < max_degree + 1:
                c[i, j] = 1

    X = np.zeros([m * n, np.sum(c).astype(np.int)])
    iterator = 0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] == 1:
                X[:, iterator] = legendre_2d(x, y, i, j).reshape(
                    m * n,
                )
                iterator = iterator + 1

    YY = np.reshape(Y, (m * n, 1))

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    surface = np.reshape(np.dot(X, theta), (m, n))

    c_new = np.zeros_like(c)
    iterator = 0
    for j in range(c.shape[1]):
        for i in range(c.shape[0]):
            if c[i, j] == 1:
                c_new[i, j] = theta[iterator]
                iterator = iterator + 1

    return surface, c_new


def legendre_fitting_nan(Y, max_degree):
    m = Y.shape[0]
    n = Y.shape[1]

    # y, x = np.mgrid[np.linspace(-1, 1, m), np.linspace(-1, 1, n)]
    y = np.linspace(-1, 1, m)
    x = np.linspace(-1, 1, n)
    c = np.zeros([max_degree + 1, max_degree + 1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if i + j < max_degree + 1:
                c[i, j] = 1

    X = np.zeros([m * n, np.sum(c).astype(np.int)])
    iterator = 0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] == 1:
                X[:, iterator] = legendre_2d_2(x, y, i, j).reshape(
                    m * n,
                )  # switched to legendre_2d_2 due to changes in legval2d
                iterator = iterator + 1

    YY = np.reshape(Y, (m * n, 1))

    # Only this part is added to regular Legendre fitting
    nan_indices = np.isnan(YY)
    nan_indices = nan_indices.reshape(
        [
            nan_indices.shape[0],
        ]
    )
    X_no_nan = X[~nan_indices]
    YY_no_nan = YY[~nan_indices]
    #####################################################

    # theta = np.dot(np.dot(np.linalg.pinv(np.dot(X_no_nan.transpose(), X_no_nan)), X_no_nan.transpose()), YY_no_nan)  # swapped variables
    theta = (np.linalg.pinv(X_no_nan.transpose() @ X_no_nan) @ X_no_nan.transpose()) @ YY_no_nan  # cleaner code

    surface = np.reshape(X @ theta, (m, n))  # surface retrieved for the original size

    c_new = np.zeros_like(c)
    iterator = 0
    for j in range(c.shape[1]):
        for i in range(c.shape[0]):
            if c[i, j] == 1:
                c_new[i, j] = theta[iterator]
                iterator = iterator + 1

    return surface, c_new


def gradient_PF(im, sigma=10):
    from scipy.ndimage import gaussian_filter, median_filter

    def filter_gradient(grad):
        grad_filt = gaussian_filter(grad, sigma=sigma)
        # grad_filt = median_filter(grad, size=sigma)
        # thr = threshold_otsu(grad_filt)
        slope = np.median(grad_filt)
        return slope

    (height, width) = im.shape
    y = np.arange(height) - height // 2
    x = np.arange(width) - width // 2
    x, y = np.meshgrid(x, y)
    grad_y, grad_x = np.gradient(im)
    y_slope = filter_gradient(grad_y)
    im -= y_slope * y
    x_slope = filter_gradient(grad_x)
    im -= x_slope * x
    return im


def plane_fitting(Y):
    m = Y.shape[0]
    n = Y.shape[1]

    X1, X2 = np.mgrid[:m, :n]

    # Regression
    X = np.hstack((np.reshape(X1, (m * n, 1)), np.reshape(X2, (m * n, 1))))
    X = np.hstack((np.ones((m * n, 1)), X))
    YY = np.reshape(Y, (m * n, 1))
    nan_indices = np.isnan(YY)
    nan_indices = nan_indices.reshape(
        [
            nan_indices.shape[0],
        ]
    )
    X_no_nan = X[~nan_indices]
    YY_no_nan = YY[~nan_indices]

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X_no_nan.transpose(), X_no_nan)), X_no_nan.transpose()), YY_no_nan)

    plane = np.reshape(np.dot(X, theta), (m, n))

    # Subtraction
    Y_sub = Y - plane
    return Y_sub, plane


def plane_fitting_svd(im):
    y = np.arange(im.shape[0]) - im.shape[0] // 2
    x = np.arange(im.shape[1]) - im.shape[1] // 2
    xx, yy = np.meshgrid(x, y)

    X = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    points = np.hstack((X, im.reshape(-1, 1)))
    points = np.transpose(np.transpose(points) - np.sum(points, 1) / im.size)

    svd = np.transpose(np.linalg.svd(points))
    theta = np.transpose(svd[0])  # normal vector of the fitted plane
    plane = np.reshape(np.dot(X, theta), (im.shape[0], im.shape[1]))
    return im - plane, plane


def plane_fitting_svd_2(im):
    x = np.linspace(-1, 1, im.shape[1])
    y = np.linspace(-1, 1, im.shape[0])
    u, s, vh = np.linalg.svd(im)
    x_coefs = np.polyfit(x, vh[0, :].transpose(), 1)
    y_coefs = np.polyfit(y, u[:, 0], 1)
    x_abr = np.asmatrix(x_coefs[0] * x + x_coefs[1])
    y_abr = np.asmatrix(y_coefs[0] * y + y_coefs[1])
    abr = (x_abr.T @ y_abr).H
    im_cor = im - abr
    return im_cor


def slope_fitting(im, mask):
    # algorithm to fit to slope in the image
    # should work better for finding slope of individual "steps" in image, instead of full image
    d_y, d_x = np.gradient(im)
    # mag = np.sqrt(np.square(d_y) + np.square(d_x))
    # sob = sobel(im)
    # thr = threshold_li(sob)

    # d_x_slope = d_y_slope = 0

    # kernel_y = gaussian_kde(np.reshape(d_y, -1))
    # kernel_x = gaussian_kde(np.reshape(d_x, -1))

    d_y[mask == True] = np.nan
    d_x[mask == True] = np.nan

    d_y_std = np.nanstd(d_y)
    d_x_std = np.nanstd(d_x)

    # hist_y_min_max = np.linspace(d_y.mean() - d_y_std, d_y.mean() + d_y_std, num=samples)
    # hist_x_min_max = np.linspace(d_x.mean() - d_x_std, d_x.mean() + d_x_std, num=samples)

    # hist_y = kernel_y(hist_y_min_max)
    # hist_x = kernel_x(hist_x_min_max)

    # d_y_slope = hist_y_min_max[np.argmax(hist_y)]
    # d_x_slope = hist_x_min_max[np.argmax(hist_x)]

    # _, d_y_slope = plane_fitting(d_y)
    # _, d_x_slope = plane_fitting(d_x)

    y = np.arange(im.shape[0]) - im.shape[0] // 2
    x = np.arange(im.shape[1]) - im.shape[1] // 2
    xx, yy = np.meshgrid(x, y)

    d_y_mean = np.nanmean(d_y)
    d_x_mean = np.nanmean(d_x)

    # d_y[sob > thr] = np.nan
    # d_x[sob > thr] = np.nan

    d_y[d_y > d_y_mean + d_y_std] = np.nan
    d_y[d_y < d_y_mean - d_y_std] = np.nan
    d_x[d_x > d_x_mean + d_x_std] = np.nan
    d_x[d_x < d_x_mean - d_x_std] = np.nan

    d_y_slope = np.nanmean(d_y)
    d_x_slope = np.nanmean(d_x)

    # d_y_slope = np.median(d_y)
    # d_x_slope = np.median(d_x)
    # d_y_slope, _ = mode(d_y, axis=None)
    # d_x_slope, _ = mode(d_x, axis=None)
    # d_y_slope = d_y_slope[0]
    # d_x_slope = d_x_slope[0]

    plane = d_x_slope * xx + d_y_slope * yy
    return plane


def wrap(im):
    return np.fmod(im, np.pi)


class LegendreOptimization:
    def __init__(self, arg, max_degree):
        if isinstance(arg, str):
            self.im = imread(arg)[1, :, :]
        elif isinstance(arg, np.ndarray):
            self.im = arg
        self.c_init = legendre_fitting(im, max_degree)[1]
        height, width = im.shape
        self.y = np.linspace(-1, 1, height)
        self.x = np.linspace(-1, 1, width)

    def merit_fun_legendre(self, c):
        abr = np.polynomial.legendre.legval2d(self.x.reshape(1, -1), self.y.reshape(-1, 1), c)
        phase = self.im - abr
        value = np.sum(np.sqrt(wrap(np.diff(phase, axis=0)[:, :-1]) ** 2 + wrap(np.diff(phase, axis=1)[:-1, :]) ** 2))
        return value

    def optimize_legendre(self):
        self.res = minimize(self.merit_fun_legendre, self.c_init, method='BFGS')
        self.c_opt = self.res.x.reshape(self.c_init.shape[0], self.c_init.shape[1])


class LegendreOptimization2:
    def __init__(self, arg, max_degree):
        if isinstance(arg, str):
            self.im = imread(arg)[1, :, :]
        elif isinstance(arg, np.ndarray):
            self.im = arg
        self.c_init = legendre_fitting(im, max_degree)[1]
        self.im = np.exp(1j * self.im)
        height, width = im.shape
        self.y = np.linspace(-1, 1, height)
        self.x = np.linspace(-1, 1, width)
        self.y = np.reshape(self.y, (-1, 1))
        self.x = np.reshape(self.x, (1, -1))

    def merit_fun_legendre(self, c):
        abr = np.polynomial.legendre.legval2d(self.x, self.y, c)
        abr = np.exp(1j * -abr)
        phase = np.multiply(self.im, abr)
        phase = np.angle(phase)
        # phase = unwrap_phase(phase)
        value = np.sum(
            np.sqrt(wrap(np.diff(phase, axis=0)[:, :-1]) ** 2 + wrap(np.diff(phase, axis=1)[:-1, :]) ** 2)
        )  # najpierw wrap czy pow2
        # value = np.max(phase) - np.min(phase)
        print(value)
        return value

    def optimize_legendre(self, show=False):
        self.res = minimize(self.merit_fun_legendre, self.c_init, method='BFGS')
        self.c_opt = self.res.x.reshape(self.c_init.shape[0], self.c_init.shape[1])
        self.surf_opt = np.polynomial.legendre.legval2d(self.x, self.y, self.c_opt)
        self.surf_opt = np.exp(1j * -self.surf_opt)
        self.phase_opt = np.multiply(self.im, self.surf_opt)
        self.phase_opt = unwrap_phase(np.angle(self.phase_opt))
        if show == True:
            plt.figure()
            plt.imshow(self.phase_opt)
            plt.colorbar()
            plt.show()


def svd_correction(im):
    x = np.linspace(-1, 1, im.shape[1])
    y = np.linspace(-1, 1, im.shape[0])
    u, s, vh = np.linalg.svd(im)
    x_coefs = np.polyfit(x, unwrap_phase(np.angle(vh[0, :])).transpose(), 2)
    y_coefs = np.polyfit(y, unwrap_phase(np.angle(u[:, 0])), 2)
    x_abr = np.asmatrix(np.exp(1j * (x_coefs[0] * x**2 + x_coefs[1] * x + x_coefs[2])))
    y_abr = np.asmatrix(np.exp(1j * (y_coefs[0] * y**2 + y_coefs[1] * y + y_coefs[2])))
    abr = np.matmul(x_abr.T, y_abr).H
    im_cor = unwrap_phase(np.angle(np.multiply(im, abr)))
    return im_cor


# def check_name(a):
#     for k, v in list(locals().iteritems()):
#         if v is a:
#             a_as_str = k
#     return a_as_str

# def subplot_abr(*args):
#     # plt.figure()
#     rows = 2
#     cols = len(args) // rows
#     img = []
#     fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
#     for i, a in enumerate(args):
#         img.append(fig.axes[i].imshow(a))
#         colorbar(img[i])
#
#
# def subplot_abr_2(ims, rows, side_lab=None, top_lab=None, title=None):
#     ph_px_size_x = 0.419932829186377
#     ph_px_size_y = 0.420781635263248
#     start_y = 265
#     stop_y = start_y + 298
#     # stop_y = 600
#     start_x = 1185
#     stop_x = start_x + 358
#     x_axis = np.linspace(0, stop_x - start_x - 1, stop_x - start_x) * ph_px_size_x
#     y_axis = np.linspace(0, stop_y - start_y - 1, stop_y - start_y) * ph_px_size_y
#     cols = len(ims) // rows
#     img = []
#     means = []
#     for i, im in enumerate(ims):
#         means.append(np.mean(im))
#         if i >= cols * (rows - 1) - 1:
#             if i == cols * rows - 1:
#                 min_val = np.min(im)
#                 max_val = np.max(im)
#
#     with plt.style.context('abr_style.mplstyle'):
#         fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
#         for i, im in enumerate(ims):
#             img.append(fig.axes[i].imshow(im - means[i], extent=[x_axis[1], x_axis[-1], y_axis[-1], y_axis[1]]))
#             if i > cols * (rows - 1) - 1:
#                 img[i].set_clim(min_val, max_val)
#             colorbar(img[i])
#         if side_lab is not None:
#             for i, im in enumerate(ims):
#                 if np.fmod(i, cols) == 0:
#                     fig.axes[i].set_ylabel(side_lab[i // cols])
#                 if i < cols:
#                     fig.axes[i].set_title(top_lab[i])
#         if title is not None:
#             fig.suptitle(title)
#     # fig.tight_layout()
#     plt.savefig("image.png", bbox_inches='tight', dpi=100)

# def mse(imageA, imageB):
#     # the 'Mean Squared Error' between the two images is the
#     # sum of the squared difference between the two images;
#     # NOTE: the two images must have the same dimension
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
#
#     # return the MSE, the lower the error, the more "similar"
#     # the two images are
#     return err
#

if __name__ == '__main__':
    from visualization import imshow_paper, plot_cross_section, subplot_multirow, tex_table, tex_multirow_table

    mpl.rcParams['image.cmap'] = 'cividis'

    # run_AME()
    # table_env = Table()
    # table_env.add_caption('test')
    # with table_env.create(Tabular('rc|cl')) as table:
    #     table.add_hline()
    #     table.add_row((1, 2, 3, 4))
    #     table.add_hline(1, 2)
    #     table.add_empty_row()
    #     table.add_row((4, 5, 6, 7))
    # table_env.generate_tex('D:\\Python\\Python3.6\\Stitching\\test2')
    images_path = 'D:\\Python\\Python3.6\\Stitching\\testset9\\Wybrane\\'
    im_name = []
    im_name.append('phase_ref017_001_x005_y016_dx.tiff')
    im_name.append('phase_ref017_001_x010_y008_dx.tiff')
    im_name.append('phase_ref017_001_x011_y015_dx.tiff')
    im_name.append('phase_ref017_001_x014_y011_dx.tiff')
    table_rows = []
    comparison_table = []
    mog_table = []
    # im_name = 'phase_ref017_001_x010_y008_dx.tiff'
    # im_name = 'phase_ref017_001_x011_y015_dx.tiff'
    # im_name = 'phase_ref017_001_x014_y011_dx.tiff'
    for i, im_name in enumerate(im_name):
        # comparison_name = 'comparison' + im_name[16:-5] + '.png'
        comparison_name = 'comparison' + im_name[16:-5] + '.svg'
        # mog_name = 'MoG' + im_name[16:-5] + '.png'
        mog_name = 'MoG' + im_name[16:-5] + '.svg'
        table_name = 'table' + im_name[16:-5]
        mog_table_name = 'mog_table' + im_name[16:-5]
        # preview_name = 'preview_' + im_name[16:-5] + '.jpg'
        preview_name = 'preview_' + im_name[16:-5] + '.png'
        cross_section_name = comparison_name[:-4] + ' - cross_section.svg'
        # save_path = 'Z:\\PS\\Artykuły\\2018\\1. ETRI - stitching i aberracje\\Double exposure - equal abr range\\'
        save_path = 'Z:\\PS\\Artykuły\\2018\\1. ETRI - stitching i aberracje\\Double exposure - final\\'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        scale = 1.5
        im = -imread(images_path + im_name)[1, :, :]
        imshow_paper(im, save_path + preview_name, col_bar=False, show_axis=False)
        im_ref = -imread("D:\\Python\\Python3.6\\Stitching\\testset9\\phase_ref017_001_x000_y008_dx.tiff")[1]
        abr_mean = imread('D:\\Python\\Python3.6\\Stitching\\testset9\\averaged_aberrations.tiff')[0, :, :]
        im_ph = np.exp(1j * im)

        im_mean = (im - abr_mean).astype(np.float64)
        im_mean -= np.mean(im_mean)
        im_double = (im - im_ref).astype(np.float64)
        im_double -= np.mean(im_double)

        # surf_zer = zernike_abr(im, D=2)
        # im_zer = im - surf_zer
        # subplot_row([im, surf_zer, im_zer])

        surf_leg, c_leg = legendre_fitting(im, 3)
        im_leg = im - surf_leg
        im_leg -= np.mean(im_leg)

        im_svd = svd_correction(im_ph)
        im_svd -= np.mean(im_svd)
        surf_svd = im - im_svd

        # leg_opt = LegendreOptimization2(im, 3)
        # leg_opt.optimize_legendre(show=False)
        # surf_opt = np.polynomial.legendre.legval2d(leg_opt.x, leg_opt.y, leg_opt.c_opt)
        # im_opt = im - surf_opt

        mog_mean = magnitude_of_gradient(im_mean)
        mog_double = magnitude_of_gradient(im_double)
        mog_leg = magnitude_of_gradient(im_leg)
        mog_svd = magnitude_of_gradient(im_svd)
        # mog_opt = magnitude_of_gradient(im_opt)

        # diff_mean = im_double - im_mean
        # diff_leg = im_double - im_leg
        # diff_svd = im_double - im_svd

        ssim_mean = compare_ssim(im_double, im_mean)
        ssim_leg = compare_ssim(im_double, im_leg)
        ssim_svd = compare_ssim(im_double, im_svd)
        ssim_row = batch_round([ssim_mean, ssim_leg, ssim_svd], 3)
        ssim_row = ['SSIM'] + ssim_row

        mse_mean = compare_mse(im_double, im_mean)
        mse_leg = compare_mse(im_double, im_leg)
        mse_svd = compare_mse(im_double, im_svd)
        mse_row = batch_round([mse_mean, mse_leg, mse_svd], 3)
        mse_row = ['MSE'] + mse_row

        psnr_mean = compare_psnr(im_double, im_mean, data_range=ims_range([im_double, im_mean]))
        psnr_leg = compare_psnr(im_double, im_leg, data_range=ims_range([im_double, im_leg]))
        psnr_svd = compare_psnr(im_double, im_svd, data_range=ims_range([im_double, im_svd]))
        psnr_row = batch_round([psnr_mean, psnr_leg, psnr_svd], 3)
        psnr_row = ['PSNR'] + psnr_row

        table_rows = []
        table_rows.append(ssim_row)
        table_rows.append(mse_row)
        table_rows.append(psnr_row)
        comparison_table.append([str(i + 1)] + [table_rows])
        # tex_multirow_table(comparison_table, save_path + 'comparison_table', ['metric'] + top_lab[1:], caption='test')

        mog_double_sum = np.sum(mog_double)
        mog_mean_sum = np.sum(mog_mean)
        mog_leg_sum = np.sum(mog_leg)
        mog_svd_sum = np.sum(mog_svd)
        mog_row = [mog_double_sum, mog_mean_sum, mog_leg_sum, mog_svd_sum]
        mog_row = batch_round(mog_row)
        mog_row = [str(i + 1)] + mog_row
        mog_table.append(mog_row)

        images_raw = [im, im, im, im, im]
        images_abr = [im_ref, abr_mean, surf_leg, surf_svd]
        images_clr = [im_double, im_mean, im_leg, im_svd]
        images_mog = [mog_double, mog_mean, mog_leg, mog_svd]
        images_all = [images_raw, images_abr, images_clr, images_mog]
        side_lab = ['Raw image', 'Aberrations', 'Cleared image', 'Variation']
        top_lab = ['DE', 'AME', 'SF', 'PCA']
        equal_mean = [True, True, True, False]
        range_mode = ['auto', 'first', 'first', 'first']
        range_clip_percentage = [None, None, None, 1]
        subplot_multirow(
            images_all,
            side_lab=side_lab,
            top_lab=top_lab,
            equal_mean=equal_mean,
            range_mode=range_mode,
            range_clip_percentage=range_clip_percentage,
            scale=scale,
            save_path=save_path + comparison_name,
            dpi=300,
            show=False,
        )

        # images_all_mog = [images_clr, images_mog]
        # side_lab_mog = ['Cleared image', 'MoG']
        # equal_mean_mog = [True, False]
        # range_mode_mog = ['first', 'first']
        # subplot_multirow(images_all_mog, side_lab=side_lab_mog, top_lab=top_lab, equal_mean=equal_mean_mog, range_mode=range_mode_mog,
        #                  scale=scale, save_path=save_path + mog_name, dpi=300, show=False)
        # tex_table(table_rows, save_path + table_name, top_lab[1:], caption=comparison_caption)
        # tex_table(mog_table, save_path + mog_table_name, top_lab, caption='Comparison of the TV of the resulting images.')

        plot_cross_section(
            images_abr,
            limits=(-2.1, 1.3),
            labels=top_lab,
            equal_mean=True,
            save_path=save_path + cross_section_name,
            show=False,
        )

    tex_multirow_table(comparison_table, save_path + 'comparison_table', ['No.', 'Metric'] + top_lab[1:])
    tex_table(mog_table, save_path + 'TV_table', ['No.'] + top_lab)
    # table_rows = np.asarray(table_rows)
    # mog_table = np.asarray(mog_table)
    test = 1

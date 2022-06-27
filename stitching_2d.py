__author__ = 'Piotr Stępień'

from scipy.io.matlab.mio import loadmat
from image_processing import set_range, set_reset_range, convert_fused_hdf5_to_array
from stitching_bs import ImageCollection, unwrapping_error_segmentation, average_phase_timelapse
from aberrations import legendre_fitting_nan, plane_fitting, gradient_PF
from scipy.ndimage import gaussian_filter, median_filter
from skimage.io import imread, imsave
from visualization import autocrop, grid_row
from scipy.io import savemat
from pathlib import Path
from importlib import reload
from segmentation import segment_cells_ws, unwrapping_error_segmentation
import matplotlib as mpl
import matplotlib.pyplot as plt
import imagej_wrapper as iwr
import gc
import numpy as np
import copy
import datasets as ds


def run():
    iwr.imagej.sj.config.add_option('-Xmx10g')  # number before 'g' is the number of GB RAM reserved for Fiji (JVM)
    mpl.cm.get_cmap('cividis')
    mpl.rcParams['image.cmap'] = 'cividis'
    plt.style.use('abr_style.mplstyle')
    # plt.ion()

    data_storage = Path('/srv/data/')
    params = ds.dataset_20211220_organoid(data_storage)
    ij = iwr.imagej.init('/srv/data/Fiji.app', headless=True)
    # ij = iwr.imagej.init('/home/piotr_gnome/Programs/fiji-linux64_2/Fiji.app/', headless=True)
    # ij = imagej.init(['net.imagej.imagej:2.1.0', 'net.preibisch:BigStitcher:0.4.1'])

    times_paths = params['paths']
    datapath = times_paths[0][0][0].parent / 'temp'
    if not datapath.exists():
        datapath.mkdir()
    dataset_xml = 'dataset.xml'
    dataset_xml_fused = str(Path(dataset_xml).stem) + '-fused.xml'
    fused_h5 = Path(datapath / (Path(dataset_xml_fused).stem + '.h5'))

    avg_abr_path = datapath / 'avr_abr.tiff'
    avg_abr_leg_path = datapath / 'avr_abr_leg.tiff'
    if not avg_abr_path.exists():
        avg_abr = average_phase_timelapse(times_paths, change_sign=params['sign'], with_preview=False)
        avg_abr_leg, _ = legendre_fitting_nan(avg_abr, 3)
        imsave(str(avg_abr_path), avg_abr)
        imsave(str(avg_abr_leg_path), avg_abr_leg)
    else:
        avg_abr = imread(str(avg_abr_path))
        avg_abr_leg = imread(str(avg_abr_leg_path))

    begin_tl = 0
    end_tl = 0
    if end_tl > begin_tl:
        times_paths = times_paths[: -(len(times_paths) - end_tl)]
    times_paths = times_paths[begin_tl:]
    tl_num = len(times_paths)

    for i in range(tl_num):
        paths = times_paths[i]
        images = ImageCollection(paths, params)
        images.description += f'_{begin_tl + i}'
        # im_full = images.arrange_grid()

        images.reverse_y()
        images.save_to_tiff_idx(with_preview=False)
        if False:
            im_full = images.arrange_grid()
            plt.imshow(im_full), plt.colorbar(), plt.show()
        abr_curr = images.return_average_aberrations()
        abr_curr_leg, _ = legendre_fitting_nan(abr_curr, 3)
        if False:
            im_raw = images.image(1, 2)
            labels = ['AME', 'FAME', 'AMEcurr', 'FAMEcurr']
            im_list = [im_raw - avg_abr, im_raw - avg_abr_leg, im_raw - abr_curr, im_raw - abr_curr_leg]
            im_list = [gradient_PF(im) for im in im_list]
            grid_row(im_list, top_lab=labels, axis=False, equal_mean=True, range_mode='first', show=True)
        images.remove_precalc_average_aberrations(avg_abr)
        images.description += '_AME'
        # images.remove_precalc_average_aberrations(avg_abr_leg); images.description += '_FAME'
        # images.remove_precalc_average_aberrations(abr_curr); images.description += '_AMEcurr'
        # images.remove_precalc_average_aberrations(abr_curr_leg); images.description += '_FAMEcurr'
        if False:
            im_part_systematic = images.arrange_grid(1, -2, 1, -2)
            preview_path = (
                data_storage / 'stitching-2d' / 'methods-paper' / ('part_systematic_' + images.description + '.mat')
            )
            if not preview_path.parent.exists():
                preview_path.parent.mkdir()
            savemat(str(preview_path), {'im': im_part_systematic})
        if False:
            im_full = images.arrange_grid()
            plt.imshow(im_full), plt.colorbar(), plt.show()

        fused_mat = fused_h5.parent / (fused_h5.stem + '_{0:0>3}'.format(i + begin_tl) + '.mat')
        name_pattern = 'phase_idx*.tiff'

        images.create_tile_config_yx(reverse_y=False, reverse_x=False)

        if False:
            images.remove_legendre_ls_2(deg=1, unwrap_err_filt=None)
        if True:
            images.gradient_slopes(sigma=10)
            # im_full_2 = images.arrange_grid(crop=True)
        if False:
            im_part_PF = images.arrange_grid(1, -2, 1, -2)
            preview_path = data_storage / 'stitching-2d' / 'methods-paper' / ('part_PF_' + images.description + '.mat')
            if not preview_path.parent.exists():
                preview_path.parent.mkdir()
            savemat(str(preview_path), {'im': im_part_PF})
        if False:
            im_full = images.arrange_grid()
            plt.imshow(im_full), plt.colorbar(), plt.show()
        if False:
            images2.remove_legendre_ls_2(deg=1, unwrap_err_filt=None)
            plt.figure(), plt.imshow(images.image(4, 1))
            plt.figure(), plt.imshow(images2.image(4, 1))
            plt.show()
            test = 1

        if False:
            im_full = images.arrange_grid()
            plt.imshow(im_full), plt.colorbar(), plt.show()

        # images.normalize_offset_5()
        # im_full = images.arrange_grid(crop=True)

        if True:
            images.minimize_offsets_error_iterative_summarized(images.offsets, 200)
            images.resave_to_tiff_with_offsets_idx()
            im_full_2 = images.arrange_grid(crop=True)

        if True:
            # images.planes_params[300] = 5
            # images.planes_params[301] = 5
            # images.planes_params[302] = 5
            # images.resave_to_tiff_with_planes_idx()
            images.minimize_offsets_tilts_error_iterative_summarized(images.planes_params, 200)
            # images.resave_to_tiff_with_planes_idx()
            im_full = images.arrange_grid(crop=True)
            test = 1
            # images.filter_planes_params(filter_fun=lambda x: median_filter(x, size=3))
            images.filter_planes_params(filter_fun=lambda x: gaussian_filter(x, sigma=0.75))
            # offsets = np.zeros(images.rows * images.cols)
            # slopes_x = np.zeros_like(offsets)
            # slopes_y = np.zeros_like(offsets)
            # for i in range(images.rows * images.cols):
            #     offsets[i] = images.planes_params[i * 3]
            #     slopes_x[i] = images.planes_params[i * 3 + 1]
            #     slopes_y[i] = images.planes_params[i * 3 + 2]
            # # offsets_im = offsets.copy().reshape((images.rows, images.cols))
            # slopes_x_im = slopes_x.copy().reshape((images.rows, images.cols))
            # slopes_y_im = slopes_y.copy().reshape((images.rows, images.cols))
            # # slopes_x_lf, _ = legendre_fitting_nan(slopes_x_im, 4)
            # # slopes_y_lf, _ = legendre_fitting_nan(slopes_y_im, 4)
            # slopes_x_lf = gaussian_filter(slopes_x_im, sigma=1.25)
            # slopes_y_lf = gaussian_filter(slopes_y_im, sigma=1.25)
            # slopes_x_hf = slopes_x_im - slopes_x_lf
            # slopes_y_hf = slopes_y_im - slopes_y_lf
            # slopes_x = slopes_x_hf.copy().reshape(-1)
            # slopes_y = slopes_y_hf.copy().reshape(-1)
            # for i in range(images.rows * images.cols):
            #     images.planes_params[i * 3] = offsets[i]
            #     images.planes_params[i * 3 + 1] = slopes_x[i]
            #     images.planes_params[i * 3 + 2] = slopes_y[i]
            images.resave_to_tiff_with_planes_idx()

        if True:
            images.minimize_offsets_error_iterative_summarized(images.offsets, 200)
            images.resave_to_tiff_with_offsets_idx()
            im_full = images.arrange_grid(crop=True)
            test = 1

        if True:
            images.normalize_offset_5()
            res = images.minimize_offsets_error(maxiter=400, downsample=5)
            images.resave_to_tiff_with_offsets_idx()

        if False:
            im_part_offsets = images.arrange_grid(1, -2, 1, -2)
            preview_path = (
                data_storage / 'stitching-2d' / 'methods-paper' / ('part_offsets_' + images.description + '.mat')
            )
            if not preview_path.parent.exists():
                preview_path.parent.mkdir()
            savemat(str(preview_path), {'im': im_part_offsets})

        if False:
            offsets = loadmat((datapath / 'offsets.mat'))['offsets'][0]
            images.resave_to_tiff_with_offsets_idx(offsets)

        # im_full = images.arrange_grid()

        if False:
            res2 = images.minimize_plane_fitting_error(
                images.plane_fitting_fitness_grad_regularized, maxiter=400, lam_edges=0, lam_tv=0.7, downsample=5
            )
            images.resave_to_tiff_with_planes_idx()
        if False:
            images.plane_fitting_grid()

        if False:
            im_full = images.arrange_grid()
            plt.imshow(im_full), plt.colorbar(), plt.show()
        if False:
            preview_path = data_storage / 'stitching-2d' / 'methods-paper' / (images.description + '.mat')
            if not preview_path.parent.exists():
                preview_path.parent.mkdir()
            savemat(str(preview_path), {'im': im_full})

        # im_full_2 = images.arrange_grid()

        px_size = images.px_size
        min_val, max_val = images.find_min_max()
        images.rescale_to_unsigned(min_o=min_val, max_o=max_val)
        iwr.define_dataset(datapath, name_pattern, px_size, savepath=None, ij=ij)
        iwr.apply_tile_configuration(datapath / 'dataset.xml', datapath / 'tileConfig.txt', ij=ij)
        iwr.calculate_pairwise_shifts(
            datapath, dataset_xml='dataset.xml', downsample_factor={'x': 1, 'y': 1, 'z': 8}, ij=ij
        )
        iwr.filter_pairwise_shifts(
            datapath, dataset_xml='dataset.xml', filter_by='total_displacement', max_displacement=20, ij=ij
        )
        iwr.optimize_globally_and_apply_shifts(
            datapath, dataset_xml='dataset.xml', relative_thr=2.5, absolute_thr=3.5, ij=ij
        )
        # iwr.icp_refinement(datapath, dataset_xml='dataset.xml', downsample_factor=None, ij=ij)
        iwr.fuse_dataset_2(datapath, dataset_xml='dataset.xml', dataset_xml_fused='dataset-fused.xml', ij=ij)
        print('Stitching finished')
        print('Converting values to phase')
        ph = convert_fused_hdf5_to_array(fused_h5, min_val, max_val, bits=16, resolution=0)
        savemat(str(fused_mat), {'ph': ph})
        ij.window().clear()
        print('Stitched {}/{}'.format(i, len(times_paths) - 1))
    test = 1

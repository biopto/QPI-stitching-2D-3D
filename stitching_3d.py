__author__ = 'Piotr Stępień'

from scipy.io.matlab.mio import savemat
from pathlib import Path
from aberrations import plane_fitting, legendre_fitting_nan
from vis import vis
from tifffile import imread, imsave
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

# import h5py
import imagej_wrapper as iwr
from utils_3d import convert_fused_hdf5_to_array, from_matlab_rec, from_matlab_pos, arrange_grid_from_pos
from stitching_bs import ImageCollection

# from tifffile import imsave
import datasets_3d
from utils import save_object, load_object, translate_offsets_to_plane_params
from utils_3d import convert_int16_to_n
import numpy as np
from scipy.ndimage import gaussian_filter


def run():
    stitch = True
    conv_to_RI = False

    if stitch:
        # os.environ['JAVA_HOME'] = 'C:\\Conda\\envs\\pyimagej\\Library\\jre\\bin\\server'
        iwr.imagej.sj.config.add_option('-Xmx25g')  # number before 'g' is the number of GB RAM reserved for Fiji (JVM)
        # ij = imagej.init()  # Initialize imagej instance
        # ij = imagej.init('C:\\Users\\BiOpTo\\Desktop\\Fiji.app', headless=True) # Initialize imagej instance
        # ij = imagej.init(['net.imagej.imagej:2.1.0', 'net.preibisch:BigStitcher:0.8.1'], headless=True)
        # ij = imagej.init(['net.imagej:imagej:2.1.0', 'net.preibisch:BigStitcher:0.8.1', 'net.imagej:imagej-legacy:0.37.4'], headless=True)
        # ij = imagej.init(['sc.fiji:fiji:2.3.1', 'net.preibisch:BigStitcher:0.8.2', 'net.imagej:imagej-legacy:0.38.1'], headless=True)
        # ij = iwr.imagej.init(['sc.fiji:fiji:2.3.1', 'net.preibisch:BigStitcher:0.8.2'], headless=True)
        # ij = iwr.imagej.init('/srv/data/Fiji.app', headless=True)
        # ij = iwr.imagej.init('D:/PS/Fiji.app', headless=True)

    for i in range(2, 3):
        # homepath, times_recs, params = datasets_3d.dataset_20211117_mevo_TL(2)
        homepath, times_recs, params = datasets_3d.dataset_20211119_organoid10V(Path('D:/PS/stitching-3d/'))
        temp_path = homepath / 'temp'
        if not temp_path.exists():
            temp_path.mkdir()

        begin_tl = 0
        times_recs = times_recs[begin_tl:]

        for i, rec_paths in enumerate(times_recs):
            t = i + begin_tl

            dataset_xml = f'dataset_{t:03d}.xml'
            dataset_xml_fused = f'dataset-fused_{t:03d}.xml'
            dataset_h5_fused = Path(dataset_xml_fused).stem + '.h5'
            pickle_name = 'images_obj_dump.pkl'
            frames_mode = 'mean'

            if stitch:
                if False:
                    x, y, z = from_matlab_pos(rec_paths)
                    paths_grid = arrange_grid_from_pos(rec_paths, x, y)
                    params['rows'] = len(paths_grid)
                    params['cols'] = len(paths_grid[0])
                    images = ImageCollection(paths_grid, params)
                    save_object(images, temp_path / pickle_name)
                # else:
                #     # images = load_object('/srv/data/stitching-3d/20211119_organoid10V/temp/images_obj_dump_0.pkl')
                #     images = load_object(temp_path / pickle_name)
                if False:
                    images = load_object(temp_path / pickle_name)
                    images.extract_tile_config()
                    images.save_to_tiff_idx_3d(overwrite=False, frames_mode=frames_mode)
                    if 'reverse_y' in params:
                        if params['reverse_y']:
                            images.reverse_y()
                    if 'reverse_x' in params:
                        if params['reverse_x']:
                            images.reverse_x()
                    save_object(images, temp_path / pickle_name)
                if True:
                    images = load_object(temp_path / pickle_name)
                    compare_methods(images, ['mean'])
                    images.resave_to_tiff_with_planes_idx()
                    im_full_390 = prepare_grid_arrangement(images, 390, 'avg', False)
                    im_full_390_corr = prepare_grid_arrangement(images, 390, 'avg', True)
                    im_full_400 = prepare_grid_arrangement(images, 400, 'avg', False)
                    im_full_400_corr = prepare_grid_arrangement(images, 400, 'avg', True)
                    im_full_mean = prepare_grid_arrangement(images, 'mean', 'avg', False)
                    im_full_mean_corr = prepare_grid_arrangement(images, 'mean', 'avg', True)
                    imsave('D:\\PS\\stitching-3d\\20211119_organoid10V\\temp\\paper\\im_full_390.tiff', im_full_390)
                    imsave(
                        'D:\\PS\\stitching-3d\\20211119_organoid10V\\temp\\paper\\im_full_390_corr.tiff',
                        im_full_390_corr,
                    )
                    imsave('D:\\PS\\stitching-3d\\20211119_organoid10V\\temp\\paper\\im_full_400.tiff', im_full_400)
                    imsave(
                        'D:\\PS\\stitching-3d\\20211119_organoid10V\\temp\\paper\\im_full_400_corr.tiff',
                        im_full_400_corr,
                    )
                    imsave('D:\\PS\\stitching-3d\\20211119_organoid10V\\temp\\paper\\im_full_mean.tiff', im_full_mean)
                    imsave(
                        'D:\\PS\\stitching-3d\\20211119_organoid10V\\temp\\paper\\im_full_mean_corr.tiff',
                        im_full_mean_corr,
                    )
                    offsets = []
                    for i in range(images.rows * images.cols):
                        offsets.append(images.planes_params[3 * i])
                    offsets = np.asarray(offsets).reshape((images.rows, -1))
                    imsave('D:\\PS\\stitching-3d\\20211119_organoid10V\\temp\\paper\\offsets_20x16.tiff', offsets)
                    images.rebuild_frames(frames_mode=400)
                    # im_full_raw = images.arrange_grid_memory(drop_overlap=True)
                    # images.rename_paths(rec_mode='offsets_1')
                    # images.rebuild_frames(frames_mode=400)
                    # im_full_offsets = images.arrange_grid_memory(drop_overlap=True)
                    # images.rename_paths(rec_mode='planes_1')
                    # images.rebuild_frames(frames_mode=400)
                    # im_full_planes = images.arrange_grid_memory(drop_overlap=True)
                    # plt.figure(), plt.imshow(convert_int16_to_n(im_full_raw)), plt.colorbar(), plt.clim((1.33, 1.36))
                    # plt.figure(), plt.imshow(convert_int16_to_n(im_full_offsets)), plt.colorbar(), plt.clim((1.33, 1.36))
                    # plt.figure(), plt.imshow(convert_int16_to_n(im_full_planes)), plt.colorbar(), plt.clim((1.33, 1.36))
                    # plt.show()
                    # images.rename_frames_paths(frames_mode='center', rename_on_disk=False)
                    # images.rebuild_frames(frames_mode=frames_mode)
                    # images.save_frames_to_disk(suffix=frames_mode)
                    # images = load_object(temp_path / 'images_obj_dump_2.pkl')
                    im_full_list = []
                    # for i in range(50, 66):
                    # images.overlap = 0.01 * i
                    images.overlap = 0.58
                    images.offsets = np.zeros_like(images.offsets)
                    images.planes_params = translate_offsets_to_plane_params(images.offsets)
                    # images.height = 530
                    # images.width = 530
                    # images.overlap = 0.5
                    # img_bg = images.frames[0][0]
                    # img_bg -= img_bg.mean()
                    # images.remove_precalc_aberrations_from_frames(img_bg)
                    # images.load_frames(suffix=frames_mode)
                    # images.rename_frames_paths(frames_mode=frames_mode, rename_on_disk=False)
                    # images.load_frames()
                    im_full_pre = images.arrange_grid_memory(
                        planes_params=images.planes_params.round(), drop_overlap=True
                    )
                    avg_abr = images.return_average_aberrations_from_frames(plane_fitting=True)
                    # avg_abr = images.return_average_aberrations_from_frames(plane_fitting=False)
                    avg_abr -= avg_abr.mean()
                    images.remove_precalc_aberrations_from_frames(avg_abr.astype(np.float64))
                    im_full_pre = images.arrange_grid_memory(drop_overlap=True)
                    images.plane_fit_frames()

                    # im_full_PF = images.arrange_grid_memory(drop_overlap=True)
                    # im_full_post = images.arrange_grid_memory(planes_params=-images.planes_params, drop_overlap=True)
                    # diff_PF = im_full_pre - im_full_PF
                    # diff_post = im_full_pre - im_full_post
                    # images.load_frames()
                    # images.remove_precalc_aberrations_from_frames(avg_abr.astype(np.float64))
                    # im_full_re = images.arrange_grid_memory(planes_params=images.planes_params, drop_overlap=True)
                    # diff_re = im_full_PF - im_full_re
                    # correlated iterative offset generation
                    # images.minimize_offsets_error_corr_iterative(images.offsets, 1, amp=1)
                    # regular iterative offset reduction
                    # images.minimize_offsets_error_iterative(images.offsets, 20, amp=1.1)
                    # images.minimize_offsets_tilts_error_iterative_summarized(images.planes_params, 300)
                    # temp_planes = images.planes_params.copy()
                    # images.filter_planes_params(filter_fun=lambda x: gaussian_filter(x, sigma=1.5), filter_offsets=False)
                    # im_full = images.arrange_grid_memory(planes_params=images.planes_params.round(), drop_overlap=True)
                    images.minimize_offsets_error_iterative_summarized(images.offsets, 200)

                    # specific to dataset_20211119_organoid10V (first image is fully background)
                    images.offsets -= images.offsets[0]

                    images.add_to_planes_params(images.offsets)
                    # offsets, planes_params = images._compare_tiffs(suffix='offsets_1')
                    # images.planes_params = translate_offsets_to_plane_params(images.offsets)
                    # images.filter_planes_params(filter_fun=lambda x: gaussian_filter(x, sigma=1.5), filter_offsets=True)
                    im_full_planes = images.arrange_grid_memory(planes_params=images.planes_params, drop_overlap=True)
                    im_full_offsets = images.arrange_grid_memory(
                        planes_params=translate_offsets_to_plane_params(images.offsets), drop_overlap=True
                    )

                    plt.figure(), plt.imshow(convert_int16_to_n(im_full_pre)), plt.colorbar(), plt.clim((1.33, 1.36))
                    plt.figure(), plt.imshow(convert_int16_to_n(im_full_offsets)), plt.colorbar(), plt.clim(
                        (1.33, 1.36)
                    )
                    plt.figure(), plt.imshow(convert_int16_to_n(im_full_planes)), plt.colorbar(), plt.clim((1.33, 1.36))
                    plt.show()

                    images.load_frames()  # removes the effect of plane_fit_frames
                    images.remove_precalc_aberrations_from_frames(avg_abr.astype(np.float64))
                    im_offsets = images.arrange_grid_memory(
                        planes_params=translate_offsets_to_plane_params(images.offsets), drop_overlap=True
                    )

                    # im_full_list.append(im_full)
                    # images.resave_to_tiff_with_offsets_idx()
                    images.resave_to_tiff_with_planes_idx()
                    pickle_name = 'images_obj_dump_offsets_4.pkl'
                    save_object(images, temp_path / pickle_name)
                    # images.plane_fit_frames()
                if False:
                    pickle_name = 'images_obj_dump_offsets_3.pkl'
                    images = load_object(temp_path / pickle_name)
                    dataset_xml = f'dataset_{t:03d}_offset.xml'
                    dataset_xml_fused = f'dataset-fused_{t:03d}_offset.xml'
                    dataset_h5_fused = Path(dataset_xml_fused).stem + '.h5'
                    params['name_pattern'] = 'REC_idx{xxx}_offsets_1.tiff'
                # p_par = translate_offsets_to_plane_params(images.offsets)
                # im_full = images.arrange_grid_memory(planes_params=p_par, drop_overlap=True)
                # plt.imshow(im_full), plt.colorbar(), plt.show()

                no_frames = len(rec_paths)
                # iwr.from_matlab_rec(homepath, rec_paths, crop_percentage=0.25)
                # from_matlab_rec(temp_path, rec_paths, crop_percentage=params['crop_percentage'])

                if False:
                    res = images.minimize_offsets_error(maxiter=100, downsample=6)
                    images.load_frames_from_disk()
                    images.planes_params = translate_offsets_to_plane_params(images.offsets)
                    im_full = images.arrange_grid_memory(planes_params=images.planes_params.round(), drop_overlap=True)
                    # plt.imshow(im_full), plt.colorbar(), plt.show()
                    # images.resave_to_tiff_with_offsets_idx()

                # Creates .xml file readable by BigStitcher
                iwr.define_dataset_tiff(
                    temp_path,
                    params['name_pattern'],
                    None,
                    0,
                    no_frames - 1,
                    zero_indexing=True,
                    dataset_xml=dataset_xml,
                    ij=ij,
                )

                # Load FoVs locations
                iwr.load_file_config(
                    temp_path, temp_path / 'tileConfig.txt', use_pixel_units=True, dataset_xml=dataset_xml, ij=ij
                )

                # Resave dataset in multiscale .h5 files
                iwr.resave_as_hdf5(temp_path, dataset_xml=dataset_xml, ij=ij)

                # Initial rough adjustment of locations based on local neighboring frames
                # iwr.calculate_pairwise_shifts(temp_path, dataset_xml=dataset_xml, ij=ij)
                macro = iwr.calculate_pairwise_shifts(
                    temp_path, dataset_xml=dataset_xml
                )  # Initial rough adjustment of locations based on local neighboring frames
                macro_path = temp_path / 'macro_pairwise'
                with open(macro_path, 'w+') as text_file:
                    text_file.write(macro)
                iwr.run_macro_2(macro_path, ij)

                iwr.filter_pairwise_shifts(
                    temp_path, dataset_xml=dataset_xml, filter_by='total_displacement', max_displacement=200, ij=ij
                )

                # Global adjustments
                iwr.optimize_globally_and_apply_shifts(
                    temp_path, relative_thr=2.5, absolute_thr=3.5, dataset_xml=dataset_xml, ij=ij
                )
                # iwr.icp_refinement(temp_path, dataset_xml=dataset_xml, ij=ij)

                # Create a single volume from multiple adjusted volumes
                iwr.fuse_dataset(temp_path, dataset_xml=dataset_xml, dataset_xml_fused=dataset_xml_fused, ij=ij)

            if conv_to_RI:
                resolution = 0
                n = convert_fused_hdf5_to_array(
                    temp_path / dataset_h5_fused, resolution=resolution
                )  # Convert 16-bit data to n values, lowest resolution for the preview
                # savemat(temp_path / f'RI_{resolution}.mat', {'RI': n})
                imsave(str(temp_path / f'RI_{resolution}.tiff'), n)  # Saving the preview as .tiff file
            test = 1

    test = 1


def compare_methods(images, methods=['center', 'mean', 'max'], planes=True, abr_second=False):
    def legendre(im, max_degree):
        abr, _ = legendre_fitting_nan(im, max_degree=max_degree)
        return im - abr

    for m in methods:
        # images.rebuild_frames(frames_mode=400)
        images.rename_frames_paths(frames_mode=m, rename_on_disk=False)
        images.load_frames()
        images.offsets = np.zeros_like(images.offsets)
        images.planes_params = translate_offsets_to_plane_params(images.offsets)
        avg_abr = images.return_average_aberrations_from_frames(plane_fitting=True)
        avg_abr -= avg_abr.mean()
        images.remove_precalc_aberrations_from_frames(avg_abr.astype(np.float64))
        if abr_second:
            abr_2 = images.frames[0][0].copy()
            abr_2 -= abr_2.mean()
            images.remove_precalc_aberrations_from_frames(abr_2)
        images.plane_fit_frames()
        if True:
            images.minimize_offsets_error_iterative_summarized(images.offsets, 200)
            # images.offsets -= images.offsets[0]
            images.add_to_planes_params(images.offsets)
            # images.filter_planes_params(filter_fun=(lambda x: gaussian_filter(x, sigma=2)), filter_offsets=True, filter_slopes=False)
            # images.filter_planes_params(filter_fun=(lambda x: legendre(x, 8)), filter_offsets=True, filter_slopes=False)
            images.subtract_from_offsets_in_planes_params(images.planes_params[0])
        else:
            images.minimize_offsets_tilts_error_iterative_summarized(images.planes_params, 300)
            images.filter_planes_params(
                filter_fun=(lambda x: gaussian_filter(x, sigma=2)), filter_offsets=True, filter_slopes=True
            )

        # specific to dataset_20211119_organoid10V (first image is fully background)
        # images.rename_frames_paths(frames_mode='center', rename_on_disk=False)
        images.rename_frames_paths(frames_mode='mean', rename_on_disk=False)
        images.load_frames()
        avg_abr = images.return_average_aberrations_from_frames(plane_fitting=True)
        avg_abr -= avg_abr.mean()
        images.remove_precalc_aberrations_from_frames(avg_abr.astype(np.float64))
        if abr_second:
            abr_2 = images.frames[0][0].copy()
            abr_2 -= abr_2.mean()
            images.remove_precalc_aberrations_from_frames(abr_2)
        if planes:
            im_full_planes = images.arrange_grid_memory(planes_params=images.planes_params, drop_overlap=True)
            plt.figure(), plt.imshow(convert_int16_to_n(im_full_planes)), plt.colorbar(), plt.clim((1.33, 1.36))
        else:
            im_full_offsets = images.arrange_grid_memory(
                planes_params=translate_offsets_to_plane_params(images.offsets), drop_overlap=True
            )
            plt.figure(), plt.imshow(convert_int16_to_n(im_full_offsets)), plt.colorbar(), plt.clim((1.33, 1.36))
    im_full_pre = images.arrange_grid_memory(drop_overlap=True)
    plt.figure(), plt.imshow(convert_int16_to_n(im_full_pre)), plt.colorbar(), plt.clim((1.33, 1.36))
    plt.show()


def prepare_grid_arrangement(images, mode=None, systematic_aberration=None, planes_params=True):
    # if z_slice is None:
    #     from tifffile import TiffFile
    #     file = TiffFile(images.paths[0][0])
    #     series = file.series[0]  # get shape and dtype of the first image series
    #     z_slice = series.shape[0] // 2
    if mode is None:
        from tifffile import TiffFile

        file = TiffFile(images.paths[0][0])
        series = file.series[0]  # get shape and dtype of the first image series
        mode = series.shape[0] // 2
    if type(mode) is not int:
        images.rename_frames_paths(frames_mode=mode, rename_on_disk=False)
        images.load_frames()
    else:
        images.rebuild_frames(frames_mode=mode)
    if systematic_aberration is None:
        systematic_aberration = np.asarray(0)
    elif systematic_aberration == 'avg':
        systematic_aberration = images.return_average_aberrations_from_frames(plane_fitting=True)
    elif type(systematic_aberration) == tuple:
        systematic_aberration = images.frames[systematic_aberration[0]][systematic_aberration[1]].copy()
    systematic_aberration -= systematic_aberration.mean()
    images.remove_precalc_aberrations_from_frames(systematic_aberration.astype(np.float64))
    if planes_params:
        im_full = images.arrange_grid_memory(planes_params=images.planes_params, drop_overlap=True)
    else:
        im_full = images.arrange_grid_memory(drop_overlap=True)
    return im_full

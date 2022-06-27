__author__ = 'Piotr Stępień'

import os
import imagej
from pathlib import Path

def define_dataset_tiff(datapath, filename, px_size=None, first_last_idx=None, number_of_tiles=None, zero_indexing=False, savepath=None, dataset_xml='dataset.xml', ij=None):
    if px_size is None:
        px_size = {'x': 1, 'y': 1, 'z': 1}
    elif type(px_size) == float:
        px_size = {'x': px_size, 'y': px_size, 'z': px_size}
    if number_of_tiles is not None:
        if zero_indexing is False:
            first_idx = 1
            last_idx = number_of_tiles
        else:
            first_idx = 0
            last_idx = number_of_tiles - 1
    if first_last_idx is None: # overrides number_of_tiles
        first_idx = first_last_idx[0]
        last_idx = first_last_idx[1]
    if savepath is None:
        savepath = datapath
    # tiles_list = ', '.join(d_list)
    # savepath_dataset = str(Path(savepath) / 'dataset')
    macro = []
    macro.append('run("Define dataset ...",')
    macro.append('"define_dataset=[Manual Loader (TIFF only, ImageJ Opener)] ')
    macro.append('project_filename={} '.format(dataset_xml))
    macro.append('multiple_timepoints=[NO (one time-point)] ')
    macro.append('multiple_channels=[NO (one channel)] ')
    macro.append('_____multiple_illumination_directions=[NO (one illumination direction)] ')
    macro.append('multiple_angles=[NO (one angle)] ')
    macro.append('multiple_tiles=[YES (one file per tile)] ')
    macro.append('image_file_directory={} '.format(datapath))
    macro.append('image_file_pattern={} '.format(filename))
    macro.append('tiles_={:d}-{:d} '.format(first_idx, last_idx))
    macro.append('calibration_type=[Same voxel-size for all views] ')
    macro.append('calibration_definition=[Load voxel-size(s) from file(s) and display for verification] ')
    macro.append('imglib2_data_container=[ArrayImg (faster)] ')
    # macro.append('show_list reconstruction_n_rec_2_margin.tiff reconstruction_n_rec_3_margin.tiff ')
    macro.append('pixel_distance_x={:.5f} '.format(px_size['x']))
    macro.append('pixel_distance_y={:.5f} '.format(px_size['y']))
    macro.append('pixel_distance_z={:.5f} '.format(px_size['z']))
    macro.append('pixel_unit=um");')
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro

def define_dataset(datapath, name_pattern, px_size=None, savepath=None, dataset_name='dataset', ij=None):
    if px_size is None:
        px_size = {'x': 1, 'y': 1, 'z': 1}
    if savepath is None:
        savepath = datapath
    # name_pattern = os.sep + name_pattern
    macro = []
    macro.append('run("Define dataset ...", "define_dataset=[Automatic Loader (Bioformats based)] project_filename={}'.format(dataset_name + '.xml'))
    macro.append('path={}'.format(str(datapath / name_pattern)))
    macro.append('exclude=10 pattern_0=Tiles pattern_1=Tiles')
    macro.append('modify_voxel_size? voxel_size_x={:.4f} voxel_size_y={:.4f} voxel_size_z={:.4f} voxel_size_unit=µm'.format(px_size['x'], px_size['y'], px_size['z']))
    macro.append('move_tiles_to_grid_(per_angle)?=[Do not move Tiles to Grid (use Metadata if available)]')
    macro.append('how_to_load_images=[Re-save as multiresolution HDF5]')
    macro.append('dataset_save_path={}'.format(str(savepath)))
    macro.append('check_stack_sizes subsampling_factors=[{ {1,1,1}, {2,2,2} }] hdf5_chunk_sizes=[{ {16,16,1}, {16,16,1} }]')
    macro.append('timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression')
    macro.append('export_path={}");'.format(str(savepath / dataset_name)))
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro

def define_dataset_idx(datapath, name_pattern, px_size, savepath=None, dataset_name='dataset', ij=None):
    if savepath is None:
        savepath = datapath
    # name_pattern = os.sep + name_pattern
    macro = []
    macro.append('run("Define dataset ...", "define_dataset=[Automatic Loader (Bioformats based)] project_filename={}'.format(dataset_name + '.xml'))
    macro.append('path={}'.format(str(datapath / name_pattern)))
    macro.append('exclude=10 pattern_0=Tiles')
    macro.append('modify_voxel_size? voxel_size_x={:.4f} voxel_size_y={:.4f} voxel_size_z=1.0000 voxel_size_unit=µm'.format(px_size['x'], px_size['y']))
    macro.append('move_tiles_to_grid_(per_angle)?=[Do not move Tiles to Grid (use Metadata if available)]')
    macro.append('how_to_load_images=[Re-save as multiresolution HDF5]')
    macro.append('dataset_save_path={}'.format(str(savepath)))
    macro.append('check_stack_sizes subsampling_factors=[{ {1,1,1}, {2,2,2} }] hdf5_chunk_sizes=[{ {16,16,1}, {16,16,1} }]')
    macro.append('timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression')
    macro.append('export_path={}");'.format(str(savepath / dataset_name)))
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro

def load_file_config(savepath, tile_config_path, dataset_xml='dataset.xml', use_pixel_units=False, ij=None):
    # use_pixel_units=False -> use actual px size, like micrometers
    # use_pixel_units=True -> use pixels as units
    dataset_xml_path = Path(savepath) / dataset_xml
    macro = []
    macro.append('run("Load TileConfiguration from File...", ')
    macro.append('"select=[{}] '.format(str(dataset_xml_path.as_posix())))
    macro.append('tileconfiguration=[{}] '.format(str(tile_config_path.as_posix())))
    if use_pixel_units:
        macro.append('use_pixel_units ')
    macro.append('keep_metadata_rotation");')
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro

# run("Load TileConfiguration from File...", "select=/home/piotr_gnome/Data/stitching_2d_data/testset9/temp/dataset.xml 
# tileconfiguration=/home/piotr_gnome/Data/stitching_2d_data/testset9/temp/tileConfig.txt keep_metadata_rotation");
def apply_tile_configuration(dataset_xml, tile_config_path, use_pixel_units=False, ij=None):
    macro = []
    macro.append('run("Load TileConfiguration from File...", "select={}'.format(dataset_xml))
    macro.append('tileconfiguration={}'.format(tile_config_path))
    if use_pixel_units:
        macro.append('use_pixel_units ')
    macro.append('keep_metadata_rotation");')
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro

def resave_as_hdf5(savepath, dataset_xml='dataset.xml', ij=None):
    # run("As HDF5 ...", "select=/home/piotr/Data/stitching_3D/resave//dataset.xml resave_angle=[All angles] resave_channel=[All channels] resave_illumination=[All illuminations] resave_tile=[All tiles] resave_timepoint=[All Timepoints] subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8} }] hdf5_chunk_sizes=[{ {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }] timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression export_path=/home/piotr/Data/stitching_3D/resave//dataset.xml");
    dataset_xml_path = Path(savepath) / dataset_xml
    macro = []
    macro.append('run("As HDF5 ...", ')
    macro.append('"select=[{}] '.format(str(dataset_xml_path.as_posix())))
    macro.append('resave_angle=[All angles] ')
    macro.append('resave_channel=[All channels] ')
    macro.append('resave_illumination=[All illuminations] ')
    macro.append('resave_tile=[All tiles] ')
    macro.append('resave_timepoint=[All Timepoints] ')
    macro.append('subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8} }] ')
    macro.append('hdf5_chunk_sizes=[{ {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }] ')
    macro.append('timepoints_per_partition=1 ')
    macro.append('setups_per_partition=0 ')
    macro.append('use_deflate_compression ')
    macro.append('export_path=[{}]");'.format(str(dataset_xml_path.as_posix())))
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro

def calculate_pairwise_shifts(savepath, dataset_xml='dataset.xml', downsample_factor=None, ij=None):
    if downsample_factor is None:
        downsample_factor = {'x': 2, 'y': 2, 'z': 4}
    dataset_xml_path = Path(savepath) / dataset_xml
    macro = []
    macro.append('run("Calculate pairwise shifts ...", ')
    macro.append('"browse={} '.format(str(dataset_xml_path)))
    macro.append('select={} '.format(str(dataset_xml_path)))
    macro.append('process_angle=[All angles] ')
    macro.append('process_channel=[All channels] ')
    macro.append('process_illumination=[All illuminations] ')
    macro.append('process_tile=[All tiles] ')
    macro.append('process_timepoint=[All Timepoints] ')
    macro.append('method=[Phase Correlation] ')
    macro.append('show_expert_grouping_options ')
    macro.append('show_expert_algorithm_parameters ')
    macro.append('how_to_treat_timepoints=[treat individually] ')
    macro.append('how_to_treat_channels=group ')
    macro.append('how_to_treat_illuminations=group ')
    macro.append('how_to_treat_angles=[treat individually] ')
    macro.append('how_to_treat_tiles=compare ')
    macro.append('downsample_in_x={} '.format(downsample_factor['x']))
    macro.append('downsample_in_y={} '.format(downsample_factor['y']))
    macro.append('downsample_in_z={} '.format(downsample_factor['z']))
    macro.append('number_of_peaks_to_check=5 ')
    macro.append('minimal_overlap=0 ')
    macro.append('subpixel_accuracy");')
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro


def filter_pairwise_shifts(savepath, dataset_xml='dataset.xml', filter_by='quality', min_r=0, max_r=1, max_shift_in_x=0, max_shift_in_y=0, max_shift_in_z=0, max_displacement=0, ij=None):
    # assert filter_by == 'quality', 'separate_displacement', 'total_displacement'
    dataset_xml_path = Path(savepath) / dataset_xml
    macro = []
    macro.append('run("Filter pairwise shifts ...", ')
    macro.append('"select={} '.format(str(dataset_xml_path)))
    if filter_by == 'quality':
        macro.append('filter_by_link_quality ')
    macro.append('min_r={} max_r={} '.format(min_r, max_r))
    if filter_by == 'separate_displacement':
        macro.append('filter_by_shift_in_each_dimension ')
    macro.append('max_shift_in_x={} max_shift_in_y={} max_shift_in_z={} '.format(max_shift_in_x, max_shift_in_y, max_shift_in_z))
    if filter_by == 'total_displacement':
        macro.append('filter_by_total_shift_magnitude ')
    macro.append('max_displacement={}");'.format(max_displacement))
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro


def optimize_globally_and_apply_shifts(savepath, dataset_xml='dataset.xml', relative_thr=2.5, absolute_thr=3.5, ij=None):
    dataset_xml_path = Path(savepath) / dataset_xml
    macro = []
    macro.append('run("Optimize globally and apply shifts ...", ')
    macro.append('"select=[{}] '.format(str(dataset_xml_path)))
    macro.append('process_angle=[All angles] ')
    macro.append('process_channel=[All channels] ')
    macro.append('process_illumination=[All illuminations] ')
    macro.append('process_tile=[All tiles] ')
    macro.append('process_timepoint=[All Timepoints] ')
    macro.append('relative={:.3f} '.format(relative_thr))  # max shift relative to neighbours, counted in px
    macro.append('absolute={:.3f} '.format(absolute_thr))  # max shift relative to initial position, counted in px
    macro.append('global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles] ')
    macro.append('fix_group_0-0");')  # first image is fixed
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro


def icp_refinement(savepath, dataset_xml='dataset.xml', downsample_factor=None, ij=None):
    if downsample_factor is None:
        downsample_factor = {'x': 8, 'y': 8, 'z': 4}
    dataset_xml_path = Path(savepath) / dataset_xml
    macro = []
    macro.append('run("ICP Refinement ...", ')
    macro.append('"select={} '.format(str(dataset_xml_path)))
    macro.append('process_angle=[All angles] ')
    macro.append('process_channel=[All channels] ')
    macro.append('process_illumination=[All illuminations] ')
    macro.append('process_tile=[All tiles] ')
    macro.append('process_timepoint=[All Timepoints] ')
    macro.append('icp_refinement_type=[Simple (tile registration)] ')
    macro.append('downsampling=[Downsampling {}/{}/{}] '.format(downsample_factor['x'], downsample_factor['y'], downsample_factor['z']))
    macro.append('interest=[Average Threshold] ')
    macro.append('icp_max_error=[Normal Adjustment (<5px)]");')
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro


def fuse_dataset(savepath, dataset_xml='dataset.xml', dataset_xml_fused='dataset-fused.xml', ij=None, save_as='hdf5', memory_usage='low'):
    # save_as – 'hdf5' or 'tiff'
    # memory_usage - 'low' or 'high'. Low is slower and requires less RAM, High is faster and requires more memory
    # run("Fuse dataset ...", "browse=/home/piotr/Data/stitching_3D/resave/dataset.xml select=/home/piotr/Data/stitching_3D/resave//dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=1 pixel_type=[32-bit floating point] interpolation=[Linear Interpolation] image=Cached interest_points_for_non_rigid=[-= Disable Non-Rigid =-] blend produce=[Each timepoint & channel] fused_image=[Save as new XML Project (HDF5)] subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8} }] hdf5_chunk_sizes=[{ {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }] timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression export_path=/home/piotr/Data/stitching_3D/resave//dataset-fuse.xml convert_32bit=[Use min/max of each image (might flicker over time)]");
    #
    dataset_xml_path = Path(savepath) / dataset_xml
    dataset_xml_fused_path = Path(savepath) / dataset_xml_fused
    dataset_h5_fused_path = dataset_xml_fused_path.parent / (str(dataset_xml_fused_path.stem) + '.h5')
    try:
        dataset_xml_fused_path.unlink()
    except OSError as e:
        print("Error: %s : %s" % (dataset_xml_fused_path, e.strerror))
    try:
        dataset_h5_fused_path.unlink()
    except OSError as e:
        print("Error: %s : %s" % (dataset_h5_fused_path, e.strerror))
    macro = []
    macro.append('run("Fuse dataset ...", ')
    # macro.append('"browse={} '.format(str(dataset_xml_path)))
    macro.append('"select=[{}] '.format(str(dataset_xml_path)))
    macro.append('process_angle=[All angles] ')
    macro.append('process_channel=[All channels] ')
    macro.append('process_illumination=[All illuminations] ')
    macro.append('process_tile=[All tiles] ')
    macro.append('process_timepoint=[All Timepoints] ')
    macro.append('bounding_box=[Currently Selected Views] ')
    macro.append('downsampling=1 ')
    macro.append('pixel_type=[16-bit unsigned integer] ')
    # macro.append('pixel_type=[32-bit floating point] ')
    macro.append('interpolation=[Linear Interpolation] ')
    if 'hdf5' in save_as and 'low' in memory_usage:
        macro.append('image=Cached ')
    elif 'tiff' in save_as:
        macro.append('image=Virtual ')
    else:
        macro.append('image=Precompute Image ')
    macro.append('interest_points_for_non_rigid=[-= Disable Non-Rigid =-] ')
    macro.append('blend produce=[Each timepoint & channel] ')
    if 'hdf5' in save_as:
        macro.append('fused_image=[Save as new XML Project (HDF5)] ')
    elif 'tiff' in save_as:
        macro.append('fused_image=[Save as new XML Project (TIFF)] ')
    macro.append('subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8} }] ')
    macro.append('hdf5_chunk_sizes=[{ {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }] ')
    macro.append('timepoints_per_partition=1 ')
    macro.append('setups_per_partition=0 ')
    macro.append('use_deflate_compression ')
    macro.append('export_path=[{}] '.format(str(dataset_xml_fused_path)))
    # macro.append('convert_32bit=[Use min/max of each image (might flicker over time)]");')
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro

def fuse_dataset_2(savepath, dataset_xml='dataset.xml', dataset_xml_fused='dataset-fused.xml', ij=None, bits=16, save_as='hdf5', memory_usage='low'):
    # run("Fuse dataset ...", "select=/home/piotr_gnome/Data/stitching_2d_data/testset9/temp/dataset.xml 
    # process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] 
    # process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=1 
    # pixel_type=[16-bit unsigned integer] interpolation=[Linear Interpolation] image=Cached 
    # interest_points_for_non_rigid=[-= Disable Non-Rigid =-] blend use produce=[Each timepoint, channel & illumination] 
    # fused_image=[Save as new XML Project (HDF5)] subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8}, {16,16,16} }] 
    # hdf5_chunk_sizes=[{ {16,16,1}, {16,16,1}, {16,16,1}, {16,16,1}, {16,16,1} }] timepoints_per_partition=1 
    # setups_per_partition=0 use_deflate_compression 
    # export_path=/home/piotr_gnome/Data/stitching_2d_data/testset9/temp/dataset-fused.xml");
    dataset_xml_path = Path(savepath) / dataset_xml
    dataset_xml_fused_path = Path(savepath) / dataset_xml_fused
    dataset_h5_fused_path = dataset_xml_fused_path.parent / (str(dataset_xml_fused_path.stem) + '.h5')
    try:
        dataset_xml_fused_path.unlink()
    except OSError as e:
        print("Error: %s : %s" % (dataset_xml_fused_path, e.strerror))
    try:
        dataset_h5_fused_path.unlink()
    except OSError as e:
        print("Error: %s : %s" % (dataset_h5_fused_path, e.strerror))
    macro = []
    macro.append('run("Fuse dataset ...", "select={}'.format(dataset_xml_path))
    macro.append('process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]')
    macro.append('process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=1')
    if bits == 16:
        macro.append('pixel_type=[16-bit unsigned integer] ')
    else:
        macro.append('pixel_type=[32-bit floating point] ')
    macro.append('interpolation=[Linear Interpolation] image=Cached')
    macro.append('interest_points_for_non_rigid=[-= Disable Non-Rigid =-] blend use produce=[Each timepoint, channel & illumination]')
    macro.append('fused_image=[Save as new XML Project (HDF5)] subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8}, {16,16,16} }]')
    macro.append('hdf5_chunk_sizes=[{ {16,16,1}, {16,16,1}, {16,16,1}, {16,16,1}, {16,16,1} }] timepoints_per_partition=1')
    macro.append('setups_per_partition=0 use_deflate_compression')
    macro.append('export_path={}");'.format(dataset_xml_fused_path))
    macro = ' '.join(macro)
    if ij is not None:
        ij.py.run_macro(macro)
    return macro

def run_macro_2(macro_path, ij):
    macro = 'runMacro("{}");'.format(str(macro_path))
    ij.py.run_macro(macro)
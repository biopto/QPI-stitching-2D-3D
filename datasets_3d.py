__author__ = 'Piotr Stępień'

from pathlib import Path
from utils_3d import extract_from_mat
import fnmatch
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def sort_natural(paths):
    paths = [str(p) for p in paths]
    paths.sort(key=natural_keys)
    return [Path(p) for p in paths]


def get_paths(homepath, dir_pattern, rec_pattern, no_timepoints=1, num=None, leading_zeros=1):
    if num is not None:
        homepath = homepath + f'{num:0{leading_zeros}d}'
    # rec_dirs = [x for x in sorted(homepath.iterdir()) if x.is_dir()]
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_paths = [
        [f for f in dir.iterdir() if f.match(rec_pattern)] for dir in rec_dirs if fnmatch.fnmatch(dir.stem, dir_pattern)
    ]
    rec_paths = [f[0] for f in rec_paths if len(f) > 0]
    times_recs = [rec_paths[i : i + len(rec_paths) // no_timepoints] for i in range(no_timepoints)]
    return times_recs


def dataset_stitching_v1():
    homepath = Path('/home/piotr/Data/Stitching_V1')
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_paths = [[f for f in dir.iterdir() if f.match('REC*.mat')] for dir in rec_dirs if dir.match('meas*')]
    rec_paths = [f[0] for f in rec_paths]
    _, params = extract_from_mat(rec_paths[0])
    params['name_pattern'] = 'meas_{xxx}_ref_015.tiff'

    return homepath, rec_paths, params


def dataset_stitching_v4():
    homepath = Path('/home/piotr/Data/Stitching_V4')
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_paths = [[f for f in dir.iterdir() if f.match('REC*.mat')] for dir in rec_dirs if dir.match('meas*')]
    rec_paths = [f[0] for f in rec_paths]
    _, params = extract_from_mat(rec_paths[0])
    params['name_pattern'] = 'meas_{xxx}_ref_038.tiff'

    return homepath, rec_paths, params


def dataset_20210119_MB_SHSY5Y():
    homepath = Path('/run/media/piotr/Data/Dane/stitching_3d/20210119_MB_SHSY5Y')
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_paths = [[f for f in dir.iterdir() if f.match('REC*.mat')] for dir in rec_dirs if dir.match('SHSY5Y*')]
    rec_paths = [f[0] for f in rec_paths]
    _, params = extract_from_mat(rec_paths[0])
    params['name_pattern'] = 'SHSY5Y_{xxx}_ref_060.tiff'
    return homepath, rec_paths, params


def dataset_20210204_shsy5y_single():
    homepath = Path('/run/media/piotr/Data/Dane/stitching_3d/2021.02.04-shsy5y/stitching')
    dir_pattern = 'shsy-5y*'
    rec_pattern = 'REC*MASK.mat'
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_paths = [
        [f for f in dir.iterdir() if f.match(rec_pattern)] for dir in rec_dirs if fnmatch.fnmatch(dir.stem, dir_pattern)
    ]
    rec_paths = [f[0] for f in rec_paths]
    _, params = extract_from_mat(rec_paths[0])
    params['name_pattern'] = 'REC_idx{xxx}.tiff'
    return homepath, rec_paths, params


def dataset_20210204_shsy5y_timelapse1():
    homepath = Path('/run/media/piotr/Data/Dane/stitching_3d/2021.02.04-shsy5y/timelapse1')
    dir_pattern = 'shsy-5y*'
    rec_pattern = 'REC*MASK.mat'
    no_timepoints = 3
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_paths = [
        [f for f in dir.iterdir() if f.match(rec_pattern)] for dir in rec_dirs if fnmatch.fnmatch(dir.stem, dir_pattern)
    ]
    rec_paths = [f[0] for f in rec_paths]
    times_recs = [rec_paths[i : i + len(rec_paths) // no_timepoints] for i in range(no_timepoints)]
    _, params = extract_from_mat(rec_paths[0])
    params['name_pattern'] = 'REC_idx{xxx}.tiff'
    return homepath, times_recs, params


def dataset_20210204_shsy5y_timelapse1_demon():
    homepath = Path('D:\\MZ\\20210204_shsy-5y\\timelapse1')
    dir_pattern = 'shsy-5y*'
    rec_pattern = 'REC*MASK.mat'
    no_timepoints = 3
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_paths = [
        [f for f in dir.iterdir() if f.match(rec_pattern)] for dir in rec_dirs if fnmatch.fnmatch(dir.stem, dir_pattern)
    ]
    rec_paths = [f[0] for f in rec_paths]
    times_recs = [rec_paths[i : i + len(rec_paths) // no_timepoints] for i in range(no_timepoints)]
    _, params = extract_from_mat(rec_paths[0])
    params['name_pattern'] = 'REC_idx{xxx}.tiff'
    return homepath, times_recs, params


def dataset_20210204_shsy5y_timelapse2_demon():
    homepath = Path('D:\\MZ\\20210204_shsy-5y\\timelapse2')
    dir_pattern = 'shsy-5y*'
    rec_pattern = 'REC*MASK.mat'
    no_timepoints = 8
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_paths = [
        [f for f in dir.iterdir() if f.match(rec_pattern)] for dir in rec_dirs if fnmatch.fnmatch(dir.stem, dir_pattern)
    ]
    rec_paths = [f[0] for f in rec_paths]
    times_recs = [rec_paths[i : i + len(rec_paths) // no_timepoints] for i in range(no_timepoints)]
    _, params = extract_from_mat(rec_paths[0])
    params['name_pattern'] = 'REC_idx{xxx}.tiff'
    return homepath, times_recs, params


def dataset_20211117_mevo_TL(num):
    homepath = Path(f'/srv/data/stitching-3d/20211117_mevo_TL/T0{num}')
    dir_pattern = 'cell_*'
    rec_pattern = 'REC*NNC.mat'
    # rec_pattern = 'REC*MASK.mat'
    no_timepoints = 1
    rec_dirs = [x for x in sorted(homepath.iterdir()) if x.is_dir()]
    rec_paths = [
        [f for f in dir.iterdir() if f.match(rec_pattern)] for dir in rec_dirs if fnmatch.fnmatch(dir.stem, dir_pattern)
    ]
    rec_paths = [f[0] for f in rec_paths if len(f) > 0]
    times_recs = [rec_paths[i : i + len(rec_paths) // no_timepoints] for i in range(no_timepoints)]
    _, params = extract_from_mat(rec_paths[0], load_rec=False)
    params['name_pattern'] = 'REC_idx{xxx}.tiff'
    params['crop_percentage'] = 0.35
    params['sign'] = 1
    params['overlap'] = 0.3
    params['px_size'] = 1
    return homepath, times_recs, params


def dataset_20211119_organoid10V(data_path):
    # import hdf5storage
    # from vis import vis
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from utils_3d import extract_from_mat, sub_mat_fov
    # path_mat = '/srv/data/stitching-3d/20211119_organoid10V/organoid10V_740_ref_540_2021-11-19_12-20-08/REC_organoid10V_740_ref_540_2021-11-19_12-20-08_DI.mat'
    # # vol = hdf5storage.loadmat(path_mat)['REC']
    # vol, _ = extract_from_mat(path_mat)
    # plt.imshow(vol.max(axis=0)), plt.colorbar(), plt.show()
    homepath = data_path / Path(f'20211119_organoid10V/')
    dir_pattern = 'organoid10V*'
    rec_pattern = 'REC*DI.mat'
    # rec_pattern = 'REC*MASK.mat'
    no_timepoints = 1
    # rec_dirs = [x for x in sorted(homepath.iterdir()) if x.is_dir()]
    rec_dirs = [x for x in homepath.iterdir() if x.is_dir()]
    rec_dirs = sort_natural(rec_dirs)
    rec_paths = [
        [f for f in dir.iterdir() if f.match(rec_pattern)] for dir in rec_dirs if fnmatch.fnmatch(dir.stem, dir_pattern)
    ]
    rec_paths = [f[0] for f in rec_paths if len(f) > 0]
    times_recs = [rec_paths[i : i + len(rec_paths) // no_timepoints] for i in range(no_timepoints)]
    _, params = extract_from_mat(rec_paths[0])
    params['name_pattern'] = 'REC_idx{xxx}.tiff'
    # params['crop_percentage'] = 0.35
    params['fov_size'] = 530
    params['fov_shift_yx'] = (335, 335)
    params['overlap'] = 0.58
    return homepath, times_recs, params

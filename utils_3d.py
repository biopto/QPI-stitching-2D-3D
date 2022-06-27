__author__ = 'Piotr Stępień'

import re
import h5py
import numpy as np
from tifffile import imsave
from pathlib import Path
from skimage.io import imread


def h5py2tiff(file_path, dst_list, index, save_path):
    # file_path – name of the h5 file to traverse
    # dst_list – list of variables to find
    # save_path - save path
    # index - index added to save_path
    def traverse(group, file):
        for dst in file[group]:
            new_dst = f'{group}/{dst}'
            if isinstance(file[new_dst], h5py._hl.dataset.Dataset):
                if dst in dst_list:
                    tmp = new_dst.replace('/', '_')
                    tmp = tmp.replace('__', '')
                    tmp = Path(save_path) / tmp
                    tmp = str(tmp) + '_' + str(index)
                    imsave(f'{tmp}.tiff', file[new_dst][()].real)
            else:
                traverse(new_dst, file)

    file = h5py.File(file_path, 'r')
    traverse('/', file)
    file.close()


class fiji_h5:
    def __init__(self, file, resolution=0):
        # if isinstance(file, str) or isinstance(file, Path):
        self.file = h5py.File(str(file), 'r')
        self.resolution = resolution

    def __getitem__(self, args):
        return self.file[f'/t00000/s00/{self.resolution}/cells'].__getitem__(args)

    def shape(self):
        return self.file[f'/t00000/s00/{self.resolution}/cells'].shape


def convert_fused_hdf5_to_array(file_path, bits=16, resolution=0, zero_to_nan=True):
    file = h5py.File(str(file_path), 'r')
    # downsampling_factor = file['/s00/resolutions'][()][resolution] # 1 means no downsampling
    ph = file[f'/t00000/s00/{resolution}/cells'][()]
    # ph = ph[0, :, :]
    ph = ph.astype(np.float)
    max_value = 0
    if bits == 16:
        max_value = np.iinfo(np.uint16).max
    # elif bits == 32:
    #     max_value = np.finfo(np.float32).max
    ph[ph < 0] += max_value
    if zero_to_nan is True:
        ph[ph == 0] = np.nan
    ph = convert_int16_to_n(ph)
    return ph


def extract_from_mat(mat_path, load_rec=True):
    if load_rec:
        variable_names = ['dxo', 'dx', 'n_immersion', 'preprocParams', 'REC']
    else:
        variable_names = ['dxo', 'dx', 'n_immersion', 'preprocParams']
    try:
        from scipy.io import loadmat

        mat = loadmat(str(mat_path), variable_names=variable_names)
    except (NotImplementedError, ValueError):
        import hdf5storage

        mat = hdf5storage.loadmat(str(mat_path), variable_names=variable_names)
    try:
        dx = mat['dxo'][0][0]
    except:
        dx = mat['dx'][0][0]
    n_imm = mat['n_immersion'][0][0]
    preproc_params = mat['preprocParams']
    for i in range(preproc_params.shape[0]):
        if 'meas_file_metadata' in preproc_params[i][0][0]:
            meas_file_metadata = preproc_params[i][1]
            for j in range(meas_file_metadata.shape[0]):
                if 'pos_x' in meas_file_metadata[j][0][0][0]:
                    pos_x = meas_file_metadata[j][0][3][0][0]
                if 'pos_y' in meas_file_metadata[j][0][0][0]:
                    pos_y = meas_file_metadata[j][0][3][0][0]
                if 'pos_z' in meas_file_metadata[j][0][0][0]:
                    pos_z = meas_file_metadata[j][0][3][0][0]
    if load_rec:
        vol = mat['REC']
        vol = np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:
        vol = None

    params = {
        'dx': dx,
        'n_imm': n_imm,
    }
    # meas_file_metadata = mat['preprocParams'][13][1]
    # pos_x = meas_file_metadata[11][0][3][0][0]
    # pos_y = meas_file_metadata[12][0][3][0][0]
    # pos_z = meas_file_metadata[13][0][3][0][0]
    params['pos_x'] = pos_x
    params['pos_y'] = pos_y
    params['pos_z'] = pos_z
    return vol, params


def from_matlab_dx(homepath, rec_paths):
    temp_path = homepath / 'temp'
    if not temp_path.exists():
        temp_path.mkdir()
    idx_min = 0
    dx_list = []
    for i, rp in enumerate(rec_paths):
        _, params = extract_from_mat(rp)
        dx_list.append(params['dx'])
    dx_avg = np.asarray(dx_list).mean()
    return dx_avg


def from_matlab_pos(paths):
    x = []
    y = []
    z = []
    for i, p in enumerate(paths):
        print(f'from_matlab_pos: {i} / {len(paths)}')
        _, params = extract_from_mat(p, load_rec=False)
        x.append(params['pos_x'])
        y.append(params['pos_y'])
        z.append(params['pos_z'])
    test = 1
    return x, y, z


def arrange_grid_from_pos(paths, x, y):
    assert len(paths) == len(x) == len(y)
    col = [0]
    row = [0]
    # paths_grid = [[paths[0]]]
    sign = 1
    for i, p in enumerate(paths[1:]):
        print(f'arrange_grid_from_pos: {i} / {len(paths[1:])}')
        i += 1
        if abs(x[i] - x[i - 1]) > abs(y[i] - y[i - 1]):
            # paths_grid[row[i-1]].append(p)
            col.append(col[i - 1] + sign)
            row.append(row[i - 1])
        else:
            col.append(col[i - 1])
            row.append(row[i - 1] + 1)
            sign = -sign
    shape = (max(row) + 1, max(col) + 1)
    paths_grid = [[0 for _ in range(shape[1])] for _ in range(shape[0])]
    for i, p in enumerate(paths):
        paths_grid[row[i]][col[i]] = p
    return paths_grid


def from_matlab_rec(temp_path, rec_paths, crop_percentage, time_point=0, dx_avg=None):
    # temp_path = homepath / 'temp'
    if not temp_path.exists():
        temp_path.mkdir()
    idx_min = 0
    # for rp in rec_paths:
    #     temp_rp = str(rp.parts[-2]).split('_')
    #     idx = int(temp_rp[1])
    #     if idx < idx_min:
    #         idx_min = idx
    params = None
    tile_config_path = temp_path / 'tileConfig.txt'
    if tile_config_path.exists():
        tile_config_path.unlink()
    with open(tile_config_path, 'w+') as f:
        f.write('dim=3')
        f.write('\n')
        for i, rp in enumerate(rec_paths):
            vol, params = extract_from_mat(rp)
            vol_16 = convert_n_to_int16(vol)
            if crop_percentage is not None:
                margin = int(round(crop_percentage / 2 * vol_16.shape[0]))
            else:
                margin = params['margin']
            vol_16 = vol_16[margin:-margin, margin:-margin, margin:-margin]
            # pos_x = params['pos_x']
            # pos_y = params['pos_y']
            # pos_z = params['pos_z']
            if dx_avg is None:
                pos_x = params['pos_x'] / params['dx']
                pos_y = params['pos_y'] / params['dx']
                pos_z = params['pos_z'] / params['dx']
            else:
                pos_x = params['pos_x'] / dx_avg
                pos_y = params['pos_y'] / dx_avg
                pos_z = params['pos_z'] / dx_avg
            # temp_rp = str(rp.parts[-2]).split('_')
            # idx = int(temp_rp[1])
            # temp_rp = '_'.join(temp_rp[:4])
            temp_rp = f'REC_idx{i:03d}'
            temp_rp = temp_path / temp_rp
            imsave(str(temp_rp) + '.tiff', vol_16)
            f.write('{};{};({:.5f}, {:.5f}, {:.5f})'.format(i - idx_min, time_point, pos_x, pos_y, pos_z))
            f.write('\n')
    with open(temp_path / 'info.txt', 'w+') as f:
        if dx_avg is None:
            f.write('dx = {:.5f} um\n'.format(params['dx']))
        else:
            f.write('dx = {:.5f} um\n'.format(dx_avg))
        f.write('n = {:.5f}\n'.format(params['n_imm']))


def recursive_folder_mat_resave(path, name_template, filename, crop_percentage=None):
    from fnmatch import fnmatch

    if isinstance(path, str):
        path = Path(path)
    n_imm = None
    dx = None
    if path.is_dir():
        for dir_path in path.iterdir():
            if fnmatch(dir_path.parts[-1], name_template) and dir_path.is_dir():
                vol, dx, n_imm = extract_from_mat(dir_path / filename)
                vol_16 = convert_n_to_int16(vol)
                if crop_percentage is not None:
                    margin = int(round(crop_percentage / 2 * vol_16.shape[0]))
                    vol_16 = vol_16[margin:-margin, margin:-margin, margin:-margin]
                imsave(str(dir_path) + '.tiff', vol_16)
    with open(path / 'info.txt', 'w+') as f:
        f.write('dx = {:.5f} um\n'.format(dx))
        f.write('n = {:.5f}\n'.format(n_imm))


def convert_to_tiff(folder_path):
    save_path = folder_path / 'resave'
    if not save_path.is_dir():
        save_path.mkdir()
    i = 0
    for p in folder_path.iterdir():
        if p.suffix == '.h5':
            h5py2tiff(p, ['n_rec'], i, save_path)
            i = i + 1


def remove_circshift(vol):
    # vol = imread(str(file))
    vol_dims = vol.shape
    vol_dims = np.asarray(vol_dims, dtype=np.int64)
    vol_dims = vol_dims // 2
    vol = np.roll(vol, tuple(vol_dims), (0, 1, 2))
    return vol


def add_margin(vol, margin=0.1):
    # margin = 0.1
    vol_dims = vol.shape
    vol_dims = np.asarray(vol_dims, dtype=np.int64)
    roll_dims = np.round(vol_dims // 2 * (1 + margin))
    roll_dims = np.asarray(roll_dims, dtype=np.int64)
    roll_dims[0] = 0
    vol_new = np.zeros((vol.shape[0], 2 * roll_dims[1], 2 * roll_dims[2]))

    vol_temp = np.roll(vol, (0, roll_dims[1], roll_dims[2]), (0, 1, 2))
    # plt.figure(), plt.imshow(vol_temp[vol_temp.shape[0]//2])
    vol_temp_crop = vol_temp[:, 0 : roll_dims[1], 0 : roll_dims[2]]
    # plt.figure(), plt.imshow(vol_temp_crop[vol_temp_crop.shape[0]//2])
    vol_new[:, 0 : roll_dims[1], 0 : roll_dims[2]] = vol_temp_crop
    # plt.figure(), plt.imshow(vol_new[vol_new.shape[0] // 2])

    vol_temp = np.roll(vol, (0, -roll_dims[1], -roll_dims[2]), (0, 1, 2))
    # plt.figure(), plt.imshow(vol_temp[vol_temp.shape[0] // 2])
    vol_temp_crop = vol_temp[:, -roll_dims[1] :, -roll_dims[2] :]
    # plt.figure(), plt.imshow(vol_temp_crop[vol_temp_crop.shape[0] // 2])
    vol_new[:, -roll_dims[1] :, -roll_dims[2] :] = vol_temp_crop
    # plt.figure(), plt.imshow(vol_new[vol_new.shape[0] // 2])

    vol_temp = np.roll(vol, (0, roll_dims[1], -roll_dims[2]), (0, 1, 2))
    # plt.figure(), plt.imshow(vol_temp[vol_temp.shape[0] // 2])
    vol_temp_crop = vol_temp[:, 0 : roll_dims[1], -roll_dims[2] :]
    # plt.figure(), plt.imshow(vol_temp_crop[vol_temp_crop.shape[0] // 2])
    vol_new[:, 0 : roll_dims[1], -roll_dims[2] :] = vol_temp_crop
    # plt.figure(), plt.imshow(vol_new[vol_new.shape[0] // 2])

    vol_temp = np.roll(vol, (0, -roll_dims[1], roll_dims[2]), (0, 1, 2))
    # plt.figure(), plt.imshow(vol_temp[vol_temp.shape[0] // 2])
    vol_temp_crop = vol_temp[:, -roll_dims[1] :, 0 : roll_dims[2]]
    # plt.figure(), plt.imshow(vol_temp_crop[vol_temp_crop.shape[0] // 2])
    vol_new[:, -roll_dims[1] :, 0 : roll_dims[2]] = vol_temp_crop
    # plt.figure(), plt.imshow(vol_new[vol_new.shape[0] // 2])
    return vol_new


def save_as_h5(filename, vol):
    with h5py.File(f'{filename}.h5', 'w') as file:
        # str_dtype = h5py.special_dtype(vlen=str)
        meas = file.create_group('RI')
        meas.create_dataset('REC', data=vol)


def convert_n_to_int16(vol):
    vol_new = vol.copy()
    vol_new = vol_new - 1
    vol_new = vol_new * (2**16 - 1)
    vol_new = np.asarray(vol_new + 0.5, dtype=np.uint16)
    return vol_new


def convert_int16_to_n(vol):
    # vol_new = np.asarray(vol.copy(), dtype=np.float16)
    vol_new = vol
    # vol_new = np.asarray(vol_new, dtype=np.float)
    vol_new = vol_new / (2**16 - 1)
    vol_new = vol_new + 1
    return vol_new


def dataset_list(x_start, x_stop, y_start, y_stop, leading_zeros=3):
    d_list = []
    for i in range(x_stop - x_start):
        for j in range(y_stop - y_start):
            elem = str(i).zfill(leading_zeros) + str(j).zfill(leading_zeros)
            d_list.append(elem)
    return d_list


def filter_tile_location_file(filename, d_list):
    # leading_zeros = len(d_list[0]) // 2
    filtered_lines = []
    filename_filtered = Path(filename).parent / (Path(filename).stem + '_' + d_list[0] + '_' + d_list[-1] + '.txt')
    with open(filename, 'r') as f:
        idx = 0
        for line in f:
            if 'dim' in line or line[0] == '#':
                filtered_lines.append(line)
            else:
                line_split = line.split(';')
                tile_no = line_split[0]
                if tile_no in d_list:
                    new_line = ';'.join([str(idx), *line_split[1:], '\n'])
                    idx = idx + 1
                    filtered_lines.append(line)
    with open(filename_filtered, 'w+') as f:
        f.writelines(filtered_lines)


def sub_mat_fov(vol, size, center_fov=None, shift_yx=None):
    vol_full = vol.shape[0]
    center = (vol_full - size) // 2
    if center_fov is None:
        center_fov = (center, center)
    if shift_yx is None:
        shift_yx = (0, 0)
    half_size = size // 2
    assert vol_full > size
    slice_y = np.asarray([0, size]) + center_fov[0] - half_size + shift_yx[0]
    slice_x = np.asarray([0, size]) + center_fov[1] - half_size + shift_yx[1]
    # slice_y = np.asarray(list(range(size))) + center + shift_yx[0]
    # slice_x = np.asarray(list(range(size))) + center + shift_yx[1]
    diff_last_y = slice_y[-1] - vol_full
    diff_last_x = slice_x[-1] - vol_full
    if diff_last_y > 0:
        slice_y -= diff_last_y
    if diff_last_x > 0:
        slice_x -= diff_last_x
    if slice_y[0] < 0:
        slice_y -= slice_y[0]
    if slice_x[0] < 0:
        slice_x -= slice_x[0]
    # shifts = [x for x in shift_yx]
    # for s in shifts:
    #     if s < center:
    #         s = center
    #     elif s > vol_full - (size + center):
    #         s = vol_full - (size + center)
    # shift_y = shifts[0]
    # shift_x = shifts[1]
    # # if (shift_yx[0] + center > vol_full) or
    # slice_y += (center + shift_y)
    # slice_x += (center + shift_x)
    return vol[:, slice_y[0] : slice_y[-1], slice_x[0] : slice_x[-1]]


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from vis import vis

    mpl.use('TkAgg')  # important for Windows
    mpl.cm.get_cmap('cividis')
    mpl.rcParams['image.cmap'] = 'cividis'
    plt.style.use('abr_style.mplstyle')

    ri_low = 1.33
    ri_high = 1.36
    resolution = 2

    vol = fiji_h5('/srv/data/stitching-3d/20211119_organoid10V/temp/dataset_offsets-f0.h5', resolution=resolution)
    vol_2 = fiji_h5(
        '/srv/data/stitching-3d/20211119_organoid10V/temp/dataset_offsets_center-f0.h5', resolution=resolution
    )
    vis(vol, fun=convert_int16_to_n, range=(ri_low, ri_high))
    plt.figure(), plt.imshow(convert_int16_to_n(vol[103, 40:880, 135:880])), plt.colorbar(), plt.clim(ri_low, ri_high)
    ims = loadmat('/srv/data/stitching-3d/20211119_organoid10V/temp/ims_full.mat')  # pre, systematic, post
    plt.figure(), plt.imshow(convert_int16_to_n(ims['pre'][:3370, 380:3250])), plt.colorbar(), plt.clim(ri_low, ri_high)
    plt.figure(), plt.imshow(convert_int16_to_n(ims['post'][:3370, 380:3250])), plt.colorbar(), plt.clim(
        ri_low, ri_high
    )
    plt.show()
    test = 1

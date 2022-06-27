__author__ = 'Piotr Stępień'

from pathlib import Path


def testset9params_big():
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.13
    first_x = 8
    last_x = 17  # 8 - 17
    first_y = 8
    last_y = 19  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    home_path = Path('/home/piotr_gnome/Data/stitching_2d_data')
    images_path = 'testset9'
    stitched_name = 'merged_fiji.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            home_path / images_path / 'phase_ref017_001_x{0:0>3}_y{1:0>3}_dx.tiff'.format(col + first_x, row + first_y)
            for row in range(rows_num)
        ]
        for col in range(cols_num)
    ]
    return (
        paths,
        overlap,
        rows_num,
        cols_num,
        stitched_name,
        images_path,
        home_path,
        percentile,
        first_x,
        last_x,
        first_y,
        last_y,
        sign,
    )


def testset9params():
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.13
    first_x = 8
    last_x = 15  # 8 - 17
    first_y = 10
    last_y = 19  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    # home_path = 'D:\\Python\\Python3.6\\Stitching\\'
    home_path = Path('/home/piotr_gnome/Data/stitching_2d_data')
    images_path = 'testset9'
    stitched_name = 'merged_fiji.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            home_path / images_path / 'phase_ref017_001_x{0:0>3}_y{1:0>3}_dx.tiff'.format(col + first_x, row + first_y)
            for row in range(rows_num)
        ]
        for col in range(cols_num)
    ]
    return (
        paths,
        overlap,
        rows_num,
        cols_num,
        stitched_name,
        images_path,
        home_path,
        percentile,
        first_x,
        last_x,
        first_y,
        last_y,
        sign,
    )


def testset9params_full():
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.13
    first_x = 0
    last_x = 25  # 8 - 17
    first_y = 0
    last_y = 30  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    home_path = Path('/home/piotr_gnome/Data/stitching_2d_data')
    images_path = 'testset9'
    stitched_name = 'merged_fiji.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            home_path / images_path / 'phase_ref017_001_x{0:0>3}_y{1:0>3}_dx.tiff'.format(col + first_x, row + first_y)
            for row in range(rows_num)
        ]
        for col in range(cols_num)
    ]
    return (
        paths,
        overlap,
        rows_num,
        cols_num,
        stitched_name,
        images_path,
        home_path,
        percentile,
        first_x,
        last_x,
        first_y,
        last_y,
        sign,
    )


def dataset_MM_test():
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.13
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    home_path = Path('/home/piotr_gnome/Data/stitching_2d_data')
    images_path = 'MM_test'
    stitched_name = 'merged_fiji.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            home_path / images_path / 'phase_ref198_0007_x{0:0>3}_y{1:0>3}.tiff'.format(col + first_x, row + first_y)
            for row in range(rows_num)
        ]
        for col in range(cols_num)
    ]
    return (
        paths,
        overlap,
        rows_num,
        cols_num,
        stitched_name,
        images_path,
        home_path,
        percentile,
        first_x,
        last_x,
        first_y,
        last_y,
        sign,
    )


def dataset_Sebastian_test_single():
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 6  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    home_path = Path('/home/piotr_gnome/Data/stitching_2d_data/Sebastian_test')
    images_path = 'test_single'
    stitched_name = 'merged_fiji.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            home_path / images_path / 'phase_ref136_0002_x{0:0>3}_y{1:0>3}.tiff'.format(col + first_x, row + first_y)
            for row in range(rows_num)
        ]
        for col in range(cols_num)
    ]
    return (
        paths,
        overlap,
        rows_num,
        cols_num,
        stitched_name,
        images_path,
        home_path,
        percentile,
        first_x,
        last_x,
        first_y,
        last_y,
        sign,
    )


def dataset_Sebastian_test_single_PW():
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    home_path = Path('/home/piotr_gnome/Data/stitching_2d_data/Sebastian_test')
    images_path = 'test_timelapse'
    stitched_name = 'merged_fiji.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            home_path / images_path / 'phase_ref136_0003_x{0:0>3}_y{1:0>3}.tiff'.format(col + first_x, row + first_y)
            for col in range(cols_num)
        ]
        for row in range(rows_num)
    ]
    return (
        paths,
        overlap,
        rows_num,
        cols_num,
        stitched_name,
        images_path,
        home_path,
        percentile,
        first_x,
        last_x,
        first_y,
        last_y,
        sign,
    )


def dataset_Sebastian_test_timelapse():
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 6  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    home_path = data_path / Path('stitching-2d/stitching_2d_data/Sebastian_test')
    images_path = 'test_single'
    stitched_name = 'merged_fiji.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            home_path / images_path / 'phase_ref136_0002_x{0:0>3}_y{1:0>3}.tiff'.format(col + first_x, row + first_y)
            for row in range(rows_num)
        ]
        for col in range(cols_num)
    ]
    return (
        paths,
        overlap,
        rows_num,
        cols_num,
        stitched_name,
        images_path,
        home_path,
        percentile,
        first_x,
        last_x,
        first_y,
        last_y,
        sign,
    )


def dataset_control_1_timelapse():
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 145
    home_path = Path('/home/piotr_gnome/Data/stitching_2d_data/Sebastian/Bez_naswietlania')
    images_path = '1_timelapse-2020-07-03T11-12'
    stitched_name = 'merged_fiji.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref136_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    return (
        paths,
        overlap,
        rows_num,
        cols_num,
        stitched_name,
        images_path,
        home_path,
        percentile,
        first_x,
        last_x,
        first_y,
        last_y,
        sign,
    )


def dataset_control_1_timelapse_PW(data_path):
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 144
    home_path = data_path / Path('stitching-2d/Naswietlanie/Pomiary_bez_naswietlania')
    images_path = '1_timelapse-2020-07-03T11-12/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'c_1'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref136_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_control_2_timelapse_PW(data_path):
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 143
    home_path = data_path / Path('stitching-2d/Naswietlanie/Pomiary_bez_naswietlania')
    images_path = '2_timelapse-2020-07-06T10-09/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'c_2'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref137_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_control_3_timelapse_PW(data_path):
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 142
    home_path = data_path / Path('stitching-2d/Naswietlanie/Pomiary_bez_naswietlania')
    images_path = '3_timelapse-2020-07-07T10-15/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'c_3'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref140_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_illuminated_1_timelapse_PW(data_path):
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 145
    home_path = data_path / Path('stitching-2d/Naswietlanie/Pomiary_naswietlane')
    images_path = '1_timelapse-2020-06-24T11-08/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'i_1'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref127_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_illuminated_2_timelapse_PW(data_path):
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 145
    home_path = data_path / Path('stitching-2d/Naswietlanie/Pomiary_naswietlane')
    images_path = '2_timelapse-2020-06-26T10-50/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'i_2'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref128_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_illuminated_2_timelapse_PW_regular_phase_retieval(data_path):
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 145
    home_path = data_path / Path('stitching-2d/Naswietlanie/Pomiary_naswietlane')
    images_path = '2_timelapse-2020-06-26T10-50/phase_reg'
    stitched_name = 'merged_fiji.tiff'
    description = 'i_2_regular'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref128_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_illuminated_3_timelapse_PW(data_path):
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 5  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 145
    home_path = data_path / Path('stitching-2d/Naswietlanie/Pomiary_naswietlane')
    images_path = '3_timelapse-2020-07-08T10-38/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'i_3'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref141_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_illuminated_SHSY5Y_0J_PW(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 6  # 8 - 17
    first_y = 1
    last_y = 7  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 145
    home_path = data_path / Path('stitching-2d/Naswietlanie/0-20J/20210202_shsy-5y_0J')
    images_path = 'timelapse-2021-02-02T12-25/phase'
    stitched_name = 'merged_fiji.tiff'
    description = '0J'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref266_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_illuminated_SHSY5Y_5J_PW(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 6  # 8 - 17
    first_y = 1
    last_y = 7  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 147
    timelapse_stop = 169
    home_path = data_path / Path('stitching-2d/Naswietlanie/0-20J/20210203_shsy-5y_5J')
    images_path = 'timelapse-2021-02-03T12-48/phase'
    stitched_name = 'merged_fiji.tiff'
    description = '5J'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref269_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_illuminated_SHSY5Y_5J_PW_2(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 7  # 8 - 17
    first_y = 1
    last_y = 7  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 4
    timelapse_stop = 144
    home_path = data_path / Path('stitching-2d/Naswietlanie/0-20J/20200831_shsy-5y_5J_2')
    images_path = 'phase'
    stitched_name = 'merged_fiji.tiff'
    description = '5J_2'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref162_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_illuminated_SHSY5Y_20J_PW(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 7  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 1
    timelapse_stop = 142
    home_path = data_path / Path('stitching-2d/Naswietlanie/0-20J/20210127_shsy-5y_20J')
    images_path = 'timelapse-2021-01-27T12-11/phase'
    stitched_name = 'merged_fiji.tiff'
    description = '20J'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref264_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_WL_20210202_shsy_5y_0J(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.27
    first_x = 1
    last_x = 6  # 8 - 17
    first_y = 1
    last_y = 7  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 1
    timelapse_stop = 2
    home_path = data_path / Path('stitching-2d/WL/20210202_shsy-5y_0J/')
    images_path = ''
    stitched_name = 'merged_fiji.tiff'
    description = '0J'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'wl_ref266_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
        'with_preview': False,
    }
    return params


def dataset_20210513_shsy_5y_5J(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 7  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 6
    timelapse_stop = 144
    home_path = data_path / Path('stitching-2d/Naswietlanie/2/20210513_5J/')
    images_path = 'timelapse-2021-05-13T12-41/phase'
    stitched_name = 'merged_fiji.tiff'
    description = '5J'
    ref_name = 'ref275.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / f'phase_{Path(ref_name).stem}_{tp:0>4}_x{col+first_x:0>3}_y{row+first_y:0>3}.tiff'
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
        'ref_name': ref_name,
        'images_path': images_path,
        'home_path': home_path,
    }
    return params


def dataset_WL_20210513_shsy_5y_5J(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.27
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 7  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 5
    timelapse_stop = 6
    home_path = data_path / Path('stitching-2d/Naswietlanie/2/20210513_5J/WL')
    images_path = ''
    stitched_name = 'merged_fiji.tiff'
    description = '5J'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'wl_ref275_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
        'with_preview': False,
    }
    return params


def dataset_WL_20210520_shsy_5y_20J(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.27
    first_x = 1
    last_x = 6  # 8 - 17
    first_y = 1
    last_y = 6  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 1
    timelapse_stop = 2
    home_path = data_path / Path('stitching-2d/Naswietlanie/2/20210520_20J')
    images_path = ''
    stitched_name = 'merged_fiji.tiff'
    description = '20J'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'wl_ref275_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
        'with_preview': False,
    }
    return params


def dataset_20210520_shsy_5y_20J(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 6  # 8 - 17
    first_y = 1
    last_y = 6  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 145
    home_path = data_path / Path('stitching-2d/Naswietlanie/2/20210520_20J/')
    images_path = 'timelapse-2021-05-20T13-08/phase'
    stitched_name = 'merged_fiji.tiff'
    description = '20J'
    ref_name = 'ref276.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / f'phase_{Path(ref_name).stem}_{tp:0>4}_x{col+first_x:0>3}_y{row+first_y:0>3}.tiff'
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
        'ref_name': ref_name,
        'images_path': images_path,
        'home_path': home_path,
    }
    return params


def dataset_HerbMed_2(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.07
    first_x = 1
    last_x = 5  # 8 - 17
    first_y = 1
    last_y = 6  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 2
    timelapse_stop = 143
    home_path = data_path / Path('stitching-2d/Herbal_medicine/')
    images_path = '2_timelapse-2021-03-23T12-39/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'HerbMed_2'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / 'phase_ref272_{0:0>4}_x{1:0>3}_y{2:0>3}.tiff'.format(tp, col + first_x, row + first_y)
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
    }
    return params


def dataset_20211202_organoid(data_path):
    images = []
    sign = 1
    percentile = 2
    # overlap = 0.35
    overlap = 0.1
    first_x = 1
    last_x = 19  # 8 - 17
    first_y = 1
    last_y = 22  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 7
    timelapse_stop = 8
    home_path = data_path / Path('/srv/data/stitching-2d/organoid/20211202_VUB-CEA_cell_phase_map_from_02-09-2021/')
    images_path = '2021-12-02_Cell_phase_map_1p5x1p5_mm/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'organoid'
    ref_name = 'ref271.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / f'phase_{Path(ref_name).stem}_{tp:0>4}_x{col+first_x:0>3}_y{row+first_y:0>3}.tiff'
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
        'ref_name': ref_name,
        'images_path': images_path,
        'home_path': home_path,
    }
    return params


def dataset_20211209_organoid(data_path):
    images = []
    sign = -1
    percentile = 2
    # overlap = 0.35
    overlap = 0.1
    first_x = 1
    last_x = 19  # 8 - 17
    first_y = 1
    last_y = 22  # 8 - 19
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    timelapse_start = 3
    timelapse_stop = 4
    home_path = data_path / Path('stitching-2d/organoid/20211209_VUB-CEA_cell_phase_map_from_02-09-2021_v2/')
    images_path = '2021-12-09_Cell_phase_map_1p5x1p5_mm/phase'
    stitched_name = 'merged_fiji.tiff'
    description = 'organoid'
    ref_name = 'ref294.tiff'
    # paths = [[images_path + 'r{:d}-c{:d}.png'.format(row, col) for col in range(cols_num)] for row in range(rows_num)]
    paths = [
        [
            [
                home_path
                / images_path
                / f'phase_{Path(ref_name).stem}_{tp:0>4}_x{col+first_x:0>3}_y{row+first_y:0>3}.tiff'
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
        'ref_name': ref_name,
        'images_path': images_path,
        'home_path': home_path,
    }
    return params


def dataset_20211220_organoid(data_path):
    sign = -1  # invert phase if necessary
    overlap = 0.1  # overlap ratio
    first_x = 1
    # phase image subset in x and y directions
    last_x = 14
    first_y = 1
    last_y = 16
    cols_num = last_x - first_x + 1
    rows_num = last_y - first_y + 1
    # timelapse range
    timelapse_start = 2  # included
    timelapse_stop = 3  # excluded
    # building paths to data
    home_path = data_path / Path('stitching-2d/organoid/2021-12-20_Cell_phase_map_1p3NA_40x/')
    images_path = 'phase'
    description = 'organoid'
    ref_name = 'ref310.tiff'
    # 2D list of paths to the images to be processed
    paths = [
        [
            [
                home_path
                / images_path
                / f'phase_{Path(ref_name).stem}_{tp:0>4}_x{col+first_x:0>3}_y{row+first_y:0>3}.tiff'
                for col in range(cols_num)
            ]
            for row in range(rows_num)
        ]
        for tp in range(timelapse_start, timelapse_stop)
    ]
    # params dict to be passed to the constructor of the ImageCollection class
    params = {
        'sign': sign,
        'overlap': overlap,
        'first_x': first_x,
        'last_x': last_x,
        'first_y': first_y,
        'last_y': last_y,
        'cols': cols_num,
        'rows': rows_num,
        'paths': paths,
        'description': description,
        'ref_name': ref_name,
        'images_path': images_path,
        'home_path': home_path,
    }
    return params

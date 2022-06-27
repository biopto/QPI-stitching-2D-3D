__author__ = 'Piotr Stępień'

# import stitching_2d
import stitching_3d
import matplotlib as mpl


if __name__ == '__main__':
    mpl.use('TkAgg')  # important for Windows
    # import imagej
    # ij = imagej.init('2.1.0')
    # print(ij.getVersion())
    # ij.dispose()
    # from utils_3d import sub_mat_fov, extract_from_mat
    # from skimage.io import imread
    # from vis import vis
    # import matplotlib.pyplot as plt
    # path_mat = ''
    # path_vol = '/srv/data/stitching-3d/20211117_mevo_TL/T02/temp/REC_idx000.tiff'
    # vol = imread(path_vol)
    # vol1 = sub_mat_fov(vol, 200, shift_xy=(50, 10))
    # vol2 = sub_mat_fov(vol, 200, shift_xy=(50, -10))
    # plt.imshow(vol1.max(axis=0)), plt.colorbar(), plt.show()
    # stitching_2d.run()
    stitching_3d.run()

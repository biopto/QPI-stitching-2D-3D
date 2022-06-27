# stitching_2D
Stitching_2D is a phase image stitching framework based on Imagej/FIJI [Grid/Collection stitching plugin](https://imagej.net/Grid/Collection_Stitching_Plugin)

## Prerequisites
- ImageJ (optional)

## Installation
fijibin (dependency) should take care of ImageJ installation, however in my experience it required some dirty workaround to work properly on Windows.

## Usage
```
images = []
sign = -1 # phase image negative
percentile = 2 # top and bottom percentile for the visually appealing preview
overlap = 0.13 # overlap ratio
first_x = 12; last_x = 17 # 8 - 17 # phase image subset in x direction
first_y = 4; last_y = 10 # 8 - 19 # phase image subset in y direction
cols_num = last_x - first_x + 1 # number of columns
rows_num = last_y - first_y + 1 # number of rows
home_path = 'D:\\Python\\Python3.6\\Stitching\\' # home path for the results
images_path = 'testset10' + '\\' # dataset path (assumed to be inside home_path)
stitched_name = 'merged_fiji.tiff' # name of the stitched result
paths = [[images_path + 'phase_ref017_003_x{0:0>3}_y{1:0>3}_dx.tiff'.format(col + first_x, row + first_y) for col in range(cols_num)] for row in range(rows_num)] # list of paths to all images to be stitched
# the rest of usage follows in stitching.py '__main__'.
```

## Author
- Piotr Stępień – _initial work_

## Cite
Piotr Stępień, Damian Korbuszewski and Małgorzata Kujawińska. ["Digital Holographic Microscopy with extended field of view using tool for generic image stitching"](https://doi.org/10.4218/etrij.2018-0499) ETRI Journal 41.1 (2019): 73-83.
# QPI-stitching-2D-3D
QPI-stitching-2D-3D is a phase image preprocessing framework to be applied before stitching. It is compatible with the [BigStitcher](https://imagej.net/plugins/bigstitcher/). It is intended to work with images acquired with our internally built hardware. Will require some minor tweaking to use with other naming conventions.

## Prerequisites for the actual stitching automation
- BigStitcher and pyimagej for starting the stitching procedure (optional, the stitching itself may be done manually)

## Installation
You may attempt to install all the dependencies with the `environment.yml` using conda:
```
conda env create -f environment.yml
```
However the environment is probably far from minimal. You may try to install packages as you test the software if you wish to minimize the number of dependencies.

## Usage
You may find examples of the usage in the `run` functions from `stitching_2D.py` and `stitching_3D.py`.

General description:
First let us define the data to be processed. This is done by adding a function to the `datasets.py` module, such as:
```python
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
```

Create the `ImageCollection` object and operate on it to modify the images stored on disk:
```python
images = ImageCollection(paths, params) # create object
images.reverse_y() # reverse the order of rows
images.save_to_tiff_idx(with_preview=False) # resave the images with proper naming conventions. with_preview omits the preview layer from the .tiff files directly from our DHM system.
images.remove_precalc_average_aberrations(avg_abr) # remove precalculated systamatic aberrations
images.gradient_slopes(sigma=10) # remove trends from individual images using the grad-PF method
images.minimize_offsets_error_iterative_summarized(images.offsets, 200) # unify baseline values using IABC method with 200 iterations
images.resave_to_tiff_with_offsets_idx() # resave the images with new baselines
im_full = images.arrange_grid(crop=True) # align images into one image (no registration), with overlaps cropped out
```

To attempt to use the BigStitcher automatically look into the `imagej_wrapper.py` module. It's best to first get to know the process manually. The macros in the module come from FIJI macro recorder, but are infused with adjustable parametrs to be passed to the functions.

To run the FIJI instance:
```python
iwr.imagej.sj.config.add_option('-Xmx10g')  # number before 'g' is the number of GB RAM reserved for Fiji (JVM)
ij = iwr.imagej.init('/srv/data/Fiji.app', headless=True) # run FIJI from manual installation location
ij = imagej.init(['net.imagej.imagej:2.1.0', 'net.preibisch:BigStitcher:0.4.1']) # run FIJI from auto-installation with specified versions of imagej and the plugins (first run may take a while)
```

After the processing:
```python
px_size = 1
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
```

## Author
- Piotr Stępień – _initial work_

## Publications
If you find the work useful, please cite one of the following papers, depending on your usage.
- Piotr Stępień, Damian Korbuszewski and Małgorzata Kujawińska. ["Digital Holographic Microscopy with extended field of view using tool for generic image stitching"](https://doi.org/10.4218/etrij.2018-0499) ETRI Journal 41.1 (2019): 73-83,
- Piotr Stępień, Wojciech Krauze, and Małgorzata Kujawińska, ["Preprocessing methods for quantitative phase image stitching"](https://doi.org/10.1364/BOE.439045) Biomed. Opt. Express 13, 1-13 (2022), 
- (Submitted) "Numerical refractive index correction for the stitching procedure in tomographic quantitative phase imaging" – Piotr Stępień, Michał Ziemczonok, Maria Baczewska, Luca Valenti, Alessandro Cherubini Elia Casirati, Małgorzata Kujawińska, Wojciech Krauze.
__author__ = 'Piotr Stępień'

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from skimage.io import imread, imsave
import copy

from skimage.util import dtype


def autocrop(image_path):
    image = imread(image_path)
    image_data = np.asarray(image)
    image_data_2 = image_data[:, :, :-1]
    image_data_bw = image_data_2.max(axis=2)
    non_empty_columns = np.where(image_data_bw.min(axis=0) < 255)[0]
    non_empty_rows = np.where(image_data_bw.min(axis=1) < 255)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    image_data_new = image_data[cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1, :]
    imsave(image_path, image_data_new)
    return


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def imshow_paper(im, name=None, col_bar=True, lims=None, px_size=None, show_axis=True):
    # path = 'Z:\\PS\\Konferencje\\2018 Speckle2018\\Manuskrypt\\'
    if px_size is None:
        ph_px_size_x = 0.419932829186377
        ph_px_size_y = 0.420781635263248
    else:
        ph_px_size_x = px_size[1]
        ph_px_size_y = px_size[0]
    start_y = 0
    stop_y = start_y + im.shape[0]
    # stop_y = 600
    start_x = 0
    stop_x = start_x + im.shape[1]
    # stop_x = 1560
    slice_paper = (slice(start_y, stop_y), slice(start_x, stop_x))
    x_axis = np.linspace(0, stop_x - start_x - 1, stop_x - start_x) * ph_px_size_x
    y_axis = np.linspace(0, stop_y - start_y - 1, stop_y - start_y) * ph_px_size_y
    plt.figure()
    plt.imshow(im[slice_paper], extent=[x_axis[1], x_axis[-1], y_axis[-1], y_axis[1]])
    if show_axis == False:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xlabel('[$\mu m$]')
        plt.ylabel('[$\mu m$]')
    if lims is not None:
        plt.clim(lims)
    if col_bar == True:
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()

    if name is not None:
        plt.savefig(name, bbox_inches=0, dpi=300)
        if name[-3:] != 'svg':
            autocrop(name)
    # autocropping
    # image = imread(name)
    # image_data = np.asarray(image)
    # image_data_2 = image_data[:,:,:-1]
    # image_data_bw = image_data_2.max(axis=2)
    # non_empty_columns = np.where(image_data_bw.min(axis=0) < 255)[0]
    # non_empty_rows = np.where(image_data_bw.min(axis=1) < 255)[0]
    # cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    #
    # image_data_new = image_data[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]
    # imsave(name, image_data_new)


def plot_paper(data, name, lims=None, px_size=None, show_axis=True):
    # path = 'Z:\\PS\\Konferencje\\2018 Speckle2018\\Manuskrypt\\'
    if px_size is None:
        ph_px_size_x = 0.419932829186377
        ph_px_size_y = 0.420781635263248
    else:
        ph_px_size_x = px_size[1]
        ph_px_size_y = px_size[0]
    # start_y = 0
    # stop_y = start_y + im.shape[0]
    # stop_y = 600
    start_x = 0
    stop_x = start_x + data.shape[0]
    # stop_x = 1560
    slice_paper = slice(start_x, stop_x)
    x_axis = np.linspace(0, stop_x - start_x - 1, stop_x - start_x) * ph_px_size_x
    # y_axis = np.linspace(0, stop_y - start_y - 1, stop_y - start_y) * ph_px_size_y
    plt.figure()
    plt.plot(data[slice_paper], extent=[x_axis[1], x_axis[-1]])
    if show_axis == False:
        plt.xticks([])
        # plt.yticks([])
    else:
        plt.xlabel('[$\mu m$]')
        plt.ylabel('[$\mu m$]')
    if lims is not None:
        plt.clim(lims)
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    autocrop(name)


def plot_cross_section(
    images,
    px_size=None,
    x_start=0,
    xlabel=None,
    ylabel=None,
    limits=None,
    labels=None,
    direction='horizontal',
    number=None,
    equal_mean=False,
    normalize=None,
    dpi=100,
    save_path=None,
    show=True,
):
    if px_size is None:
        px_size = 1
    elif px_size == 'DHM x50':
        px_size = 0.419932829186377
    if equal_mean == True:
        means = [np.mean(im) for im in images]
    else:
        means = [0 for im in images]
    plt.figure()
    for i, im in enumerate(images):
        im -= means[i]
        if direction == 'horizontal':
            if number is None:
                number = round(im.shape[0] * 0.5)
            else:
                if number < 1 and number != 0:
                    number = round(im.shape[0] * number)
            line = np.ndarray.flatten(im[number, :])
        elif direction == 'vertical':
            if number is None:
                number = round(im.shape[1] * 0.5)
            else:
                if number < 1 and number != 0:
                    number = round(im.shape[1] * number)
            line = np.ndarray.flatten(im[:, number])
        if normalize == 'mean':
            line -= np.mean(line)
        elif normalize == 'min':
            line -= np.min(line)
        stop_x = x_start + line.shape[0]
        x_axis = np.linspace(0, stop_x - x_start - 1, stop_x - x_start) * px_size
        plt.plot(x_axis, line)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if limits is not None:
        plt.ylim(limits)
    if labels is not None:
        plt.legend(labels, loc='lower right')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        if save_path[-3:] != 'svg':
            autocrop(save_path)
    if show is True:
        plt.show()


def grid_row(
    images,
    fig=None,
    number_in_fig=0,
    side_lab=None,
    top_lab=None,
    title=None,
    axis=True,
    equal_mean=False,
    equal_pix_loc=None,
    mean_offset=0,
    range_mode='auto',
    range=None,
    range_clip_percentage=None,
    save_path=None,
    plot=False,
    cut_im=0.5,
    show=False,
    savefig_kwargs={'dpi': 200},
):
    from mpl_toolkits.axes_grid1 import ImageGrid

    rows = 1
    cols = len(images)
    line_idx = round(cut_im * images[0].shape[0])
    line = images[0][line_idx, :]
    img = []
    if equal_mean == True:
        if plot is False:
            means = [np.mean(im) + mean_offset for im in images]
        else:
            means = [np.mean(im[line_idx, :]) + mean_offset for im in images]
    else:
        if equal_pix_loc is None:
            means = [mean_offset for im in images]
        else:
            equal_pix_loc = np.asarray(equal_pix_loc)
            px_idx = [np.asarray(np.asarray(im.shape) * equal_pix_loc, dtype=np.int) for im in images]
            means = [im[px_idx[i][0], px_idx[i][1]] for i, im in enumerate(images)]
    if range_mode == 'first':
        if range_clip_percentage is None:
            if range is None:
                min_val = np.min(images[0] - means[0])
                max_val = np.max(images[0] - means[0])
            else:
                min_val = range[0]
                max_val = range[1]
        else:
            min_val = np.percentile(images[0], range_clip_percentage)
            max_val = np.percentile(images[0], 100 - range_clip_percentage)
        if plot is True:
            if range is None:
                min_val = np.min(line) - means[0]
                max_val = np.max(line) - means[0]
            else:
                min_val = range[0]
                max_val = range[1]
    with plt.style.context('abr_style.mplstyle'):
        if fig is None:
            fig = plt.figure()
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(rows, cols),
            axes_pad=0.1,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.1,
        )
        for i, a in enumerate(images):
            im_fin = a - means[i]
            if plot == False:
                img.append(grid[i + number_in_fig].imshow(im_fin))
                if axis is False:
                    grid[i + number_in_fig].axis('off')
            else:
                center_line = round(cut_im * im_fin.shape[0])
                img.append(grid[i + number_in_fig].plot(im_fin[center_line, :]))
                if axis is False:
                    grid[i + number_in_fig].axis('off')
            if i == 0 and side_lab is not None:
                grid[i + number_in_fig].set_ylabel(side_lab)
            if range_mode == 'first':
                if plot is False:
                    img[i].set_clim(min_val, max_val)
                else:
                    grid[i + number_in_fig].set_ylim([min_val, max_val])
            elif range_mode == 'auto' and range_clip_percentage is not None:
                min_val = np.percentile(images[i], range_clip_percentage)
                max_val = np.percentile(images[i], 100 - range_clip_percentage)
            if (type(a[0, 0]) is not np.bool_) and plot is False:
                if i == cols - 1:
                    plt.colorbar(img[i], cax=grid.cbar_axes[0])
            if top_lab is not None:
                grid[i + number_in_fig].set_title(top_lab[i])
        if title is not None:
            grid.set_title(title)
    # fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', **savefig_kwargs)
    if show == True:
        plt.show()
    return img


def subplot_row(
    images,
    fig=None,
    number_in_fig=0,
    side_lab=None,
    top_lab=None,
    title=None,
    axis=True,
    equal_mean=False,
    equal_pix_loc=None,
    mean_offset=0,
    range_mode='auto',
    range=None,
    range_clip_percentage=None,
    save_path=None,
    plot=False,
    cross_sec_location=0.5,
    show=False,
    savefig_kwargs={'dpi': 200},
):
    rows = 1
    cols = len(images)
    line_idx = round(cross_sec_location * images[0].shape[0])
    line = images[0][line_idx, :]
    img = []
    if equal_mean == True:
        if plot is False:
            means = [np.mean(im) + mean_offset for im in images]
        else:
            means = [np.mean(im[line_idx, :]) + mean_offset for im in images]
    else:
        if equal_pix_loc is None:
            means = [mean_offset for im in images]
        else:
            equal_pix_loc = np.asarray(equal_pix_loc)
            px_idx = [np.asarray(np.asarray(im.shape) * equal_pix_loc, dtype=np.int) for im in images]
            means = [im[px_idx[i][0], px_idx[i][1]] for i, im in enumerate(images)]
    if range_mode == 'first':
        if range_clip_percentage is None:
            if range is None:
                min_val = np.min(images[0] - means[0])
                max_val = np.max(images[0] - means[0])
            else:
                min_val = range[0]
                max_val = range[1]
        else:
            min_val = np.percentile(images[0], range_clip_percentage)
            max_val = np.percentile(images[0], 100 - range_clip_percentage)
        if plot is True:
            if range is None:
                min_val = np.min(line) - means[0]
                max_val = np.max(line) - means[0]
            else:
                min_val = range[0]
                max_val = range[1]
    with plt.style.context('abr_style.mplstyle'):
        if fig is None:
            fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, tight_layout=True)
        for i, a in enumerate(images):
            im_fin = a - means[i]
            if plot == False:
                img.append(fig.axes[i + number_in_fig].imshow(im_fin))
                if axis is False:
                    ax[i + number_in_fig].axis('off')
            else:
                center_line = round(cross_sec_location * im_fin.shape[0])
                img.append(fig.axes[i + number_in_fig].plot(im_fin[center_line, :]))
                if axis is False:
                    ax[i + number_in_fig].axis('off')
            if i == 0 and side_lab is not None:
                fig.axes[i + number_in_fig].set_ylabel(side_lab)
            if range_mode == 'first':
                if plot is False:
                    img[i].set_clim(min_val, max_val)
                else:
                    fig.axes[i + number_in_fig].set_ylim([min_val, max_val])
            elif range_mode == 'auto' and range_clip_percentage is not None:
                min_val = np.percentile(images[i], range_clip_percentage)
                max_val = np.percentile(images[i], 100 - range_clip_percentage)
            if (type(a[0, 0]) is not np.bool_) and plot is False:
                colorbar(img[i])
            if top_lab is not None:
                fig.axes[i + number_in_fig].set_title(top_lab[i])
        if title is not None:
            fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', **savefig_kwargs)
    if show == True:
        plt.show()
    return img


def subplot_multirow(
    images,
    side_lab=None,
    top_lab=None,
    title=None,
    equal_mean=None,
    range_mode=None,
    range_clip_percentage=None,
    scale=1,
    save_path=None,
    dpi=100,
    plot=False,
    show=True,
):
    rows = len(images)
    cols = len(images[0])
    height, width = images[0][0].shape
    aspect_ratio = (rows * height) / (cols * width)
    fig_width = 8.3 * scale  # inches
    fig_height = fig_width * aspect_ratio
    if equal_mean is None:
        equal_mean = [None for i in range(rows)]
    if range_mode is None:
        range_mode = [None for i in range(rows)]
    if range_clip_percentage is None:
        range_clip_percentage = [None for i in range(rows)]
    with plt.style.context('abr_style.mplstyle'):
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, tight_layout=True, figsize=(fig_width, fig_height))
        for i, row in enumerate(images):
            num_in_fig = i * cols
            if i == 0:
                subplot_row(
                    images[i],
                    fig=fig,
                    number_in_fig=num_in_fig,
                    side_lab=side_lab[i],
                    top_lab=top_lab,
                    equal_mean=equal_mean[i],
                    range_mode=range_mode[i],
                    range_clip_percentage=range_clip_percentage[i],
                )
            else:
                subplot_row(
                    images[i],
                    fig=fig,
                    number_in_fig=num_in_fig,
                    side_lab=side_lab[i],
                    top_lab=None,
                    equal_mean=equal_mean[i],
                    range_mode=range_mode[i],
                    range_clip_percentage=range_clip_percentage[i],
                )
        if title is not None:
            fig.suptitle(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    if show == True:
        plt.show()


def tex_table(rows, save_path, headers=None, params=None, full=False, caption=None):
    from pylatex import Center, Tabular, Table

    if params is None:
        params = ''
        for i in range(len(rows[0])):
            params += 'c|'
        params = params[:-1]
        # params += '|'
    # centered_env = Center()
    # with centered_env.create(Table()) as table_env:
    table_env = Table()
    if caption is not None:
        table_env.add_caption(caption)
    with table_env.create(Center()) as centered_env:
        with centered_env.create(Tabular(params, booktabs=False)) as table:
            table.add_hline()
            if headers is not None:
                if len(rows[0]) - len(headers) == 1:
                    headers = [''] + headers
                table.add_row(headers)
                table.add_hline()
                table.add_hline()
                # table.end_table_header()
                # table.add_hline()
            for r in rows:
                table.add_row(r)
            table.add_hline()
        if full is False:
            centered_env.generate_tex(save_path)
    if full is True:
        table_env.generate_tex(save_path)


def tex_multirow_table(rows, save_path, headers=None, params=None, full=False, caption=None):
    from pylatex import Center, Tabular, Table, MultiRow

    width = len(rows[0][1][0]) + 1
    height = len(rows[0][1])
    if params is None:
        params = ''
        for i in range(width):
            params += 'c|'
        params = params[:-1]
        # params += '|'
    centered_env = Center()
    # with centered_env.create(Table()) as table_env:
    table_env = Table()
    if caption is not None:
        table_env.add_caption(caption)
    with table_env.create(Center()) as centered_env:
        with centered_env.create(Tabular(params, booktabs=False)) as table:
            table.add_hline()
            if headers is not None:
                if width - len(headers) == 1:
                    headers = [''] + headers
                table.add_row(headers)
                table.add_hline()
                table.add_hline()
                # table.end_table_header()
                # table.add_hline()
            for mr in rows:
                table.add_row((MultiRow(height, data=mr[0]), *mr[1][0]))
                for r in mr[1][1:]:
                    # table.add_hline(2, width)
                    table.add_row([''] + r)
                table.add_hline()
        if full is False:
            centered_env.generate_tex(save_path)
    if full is True:
        table_env.generate_tex(save_path)


def imshow_no_frames(im, outfile, lims=None):
    fig, ax = plt.subplots()
    ax.imshow(im)

    if lims is not None:
        # plt.clim(lims)
        ax.get_images()[0].set_clim(lims)

    fig.patch.set_visible(False)
    ax.axis('off')

    # with open(outfile, 'w') as outfile:
    fig.savefig(outfile, dpi=600)


def imshow_timelapse(im, px_size, lims=None, outfile=None, colorbar=False, colormap='cividis', dpi=600, show=False):
    start_y = 0
    stop_y = start_y + im.shape[0]
    start_x = 0
    stop_x = start_x + im.shape[1]
    x_axis = np.linspace(0, stop_x - start_x - 1, stop_x - start_x) * px_size[1]
    y_axis = np.linspace(0, stop_y - start_y - 1, stop_y - start_y) * px_size[0]

    fig = plt.figure()
    plt.imshow(im, extent=[x_axis[1], x_axis[-1], y_axis[-1], y_axis[1]])

    plt.set_cmap(colormap)

    if colorbar is True:
        # # create an axes on the right side of ax. The width of cax will be 5%
        # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # ax = plt.gca()
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(cax=cax)
        cbar = plt.colorbar()
        cbar.set_label('[rad]')

    plt.ylabel('[$\mu m$]')
    plt.xlabel('[$\mu m$]')

    if lims is not None:
        plt.clim(lims)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches=0, dpi=dpi)
        if outfile[-3:] != 'svg':
            autocrop(outfile)

    if show is True:
        plt.show()
    else:
        plt.close(fig)

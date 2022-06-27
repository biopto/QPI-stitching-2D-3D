__author__ = 'Piotr Stępień'

# Given im1 and im2 images stored as numpy arrays ...

# import pyelastix
# from scipy.io import loadmat, savemat
# import numpy as np
import matplotlib.pyplot as plt

# from skimage.filters.thresholding import threshold_otsu
# from skimage.morphology import remove_small_objects
# from skimage.measure import label, regionprops
# from scipy import ndimage as ndi


class IndexTracker:
    def __init__(self, X, sliced_axis=2, axis_labels=['y', 'x', 'z'], fun=None, range=None):
        self.fig, self.ax = plt.subplots(1, 1)

        self.axis_labels = axis_labels
        self.current_axis = int(sliced_axis)
        self.current_label = axis_labels[self.current_axis]

        self.X = X
        if fun is None:
            fun = lambda x: x
        self.fun = fun
        self.range = range

        try:
            self.current_axis_size = self.X.shape[self.current_axis]
        except:
            self.current_axis_size = self.X.shape()[self.current_axis]
        # self.X_rot = self.X.copy()
        # rows, cols, self.slices = X.shape
        try:
            self.ind = self.X.shape[self.current_axis] // 2
        except:
            self.ind = self.X.shape()[self.current_axis] // 2

        self.y_mouse_prev = None

        temp_x = self.X[:, :, self.ind]
        temp_x = fun(temp_x)

        self.im = self.ax.imshow(temp_x)
        if range is not None:
            self.im.set_clim(range[0], range[1])
        # self.im = self.ax.imshow(self.X_rot[:, :, self.ind])
        self.fig.colorbar(self.im, ax=self.ax)

        self.update()
        self.ax.set_title('Mouse drag up/down to select slice')

    def onmousemove(self, event):
        if event.x is not None and event.y is not None:
            if str(event.button) == 'MouseButton.LEFT':
                steps = int(round((event.y - self.y_mouse_prev) / 2))
                # if steps > 0:
                #     print('up')
                # if steps < 0:
                #     print('down')
                self.ind = (self.ind + steps) % self.current_axis_size
                self.update()
                # if self.y_mouse_prev > event.y:
                #     # print('up')
                #     if self.ind < self.current_axis_size:
                #         self.ind = (self.ind + 1) % self.current_axis_size
                #         self.update()
                # if self.y_mouse_prev < event.y:
                #     # print('down')
                #     if self.ind > 0:
                #         self.ind = (self.ind - 1) % self.current_axis_size
                #         self.update()
            self.y_mouse_prev = event.y
        else:
            self.y_mouse_prev = None

    def onrightmouseclick(self, event):
        if str(event.button) == 'MouseButton.RIGHT':
            self.current_axis = int(self.current_axis + 1)
            self.current_axis = self.current_axis % 3
            self.current_label = self.axis_labels[self.current_axis]
            try:
                self.current_axis_size = self.X.shape[self.current_axis]
            except:
                self.current_axis_size = self.X.shape()[self.current_axis]
            self.ind = self.ind % self.current_axis_size
            self.ax.clear()
            if self.current_axis == 0:
                X = self.fun(self.X[self.ind])
            elif self.current_axis == 1:
                X = self.fun(self.X[:, self.ind, :])
            else:
                X = self.fun(self.X[..., self.ind])
            self.im = self.ax.imshow(X)
            if range is not None:
                self.im.set_clim(self.range[0], self.range[1])
            # self.X_rot = self.X.copy()
            # if self.current_axis == 0:
            #     self.X_rot = np.rot90(self.X_rot, axes=[0, 2])
            # elif self.current_axis == 1:
            #     self.X_rot = np.rot90(self.X_rot, axes=[1, 2])
            #     self.X_rot = np.rot90(self.X_rot, axes=[0, 1], k=2)
            #     self.X_rot = np.rot90(self.X_rot, axes=[0, 2], k=2)
            self.update()
            # plt.colorbar()

    def update(self):
        if self.current_axis == 0:
            X = self.X[self.ind]
        elif self.current_axis == 1:
            X = self.X[:, self.ind, :]
        else:
            X = self.X[..., self.ind]
        X = self.fun(X)
        self.im.set_data(X)
        # self.im.set_data(self.X_rot[:, :, self.ind])
        self.ax.set_title(self.current_label + ' = ' + str(self.ind) + ' / ' + str(self.current_axis_size))
        self.im.axes.figure.canvas.draw()


def vis(vol, sliced_axis=2, axis_labels=['y', 'x', 'z'], fun=None, range=None):
    # import matplotlib.style as mplstyle
    # mplstyle.use('fast')
    tracker = IndexTracker(vol, sliced_axis=sliced_axis, axis_labels=axis_labels, fun=fun, range=range)
    tracker.fig.canvas.mpl_connect('motion_notify_event', tracker.onmousemove)
    tracker.fig.canvas.mpl_connect('button_press_event', tracker.onrightmouseclick)
    plt.show(block=True)


if __name__ == "__main__":
    test = 1

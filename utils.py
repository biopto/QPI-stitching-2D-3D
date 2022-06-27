__author__ = 'Piotr Stępień'

import pickle
import numpy as np


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, -1)


def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)


def translate_offsets_to_plane_params(offsets):
    dims = offsets.shape[0]
    plane_params = np.zeros(dims * 3)
    for i in range(dims):
        idx3 = i * 3
        plane_params[idx3] = offsets[i]
    return plane_params

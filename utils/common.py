import numpy as np


def reshape_and_normalize(array, new_shape, scale):
    reshaped_list = np.reshape(array, new_shape)
    return reshaped_list / float(scale)

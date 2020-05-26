from evaluation_framework.utils.data_structure_utils import is_nested_list

import numpy as np


def write_memmap(filepath, dtype, shape, array):

    writable_memmap = np.memmap(filepath, dtype=dtype, mode="w+", shape=shape)
    writable_memmap[:] = array[:]
    del writable_memmap

def read_memmap(filepath, dtype, shape, idx=None):

    readonly_memmap = np.memmap(filepath, dtype=dtype, mode="r", shape=shape)

    if idx is None:
        array = readonly_memmap[:]
    else:
        if is_nested_list(idx):
            array = readonly_memmap[tuple(idx)]
        else:
            array = readonly_memmap[idx]

    del readonly_memmap
    return array


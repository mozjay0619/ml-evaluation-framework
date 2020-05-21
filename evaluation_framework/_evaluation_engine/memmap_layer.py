from evaluation_framework.utils.decorator_utils import failed_method_retry

import numpy as np
import os
import pickle


def translate_hdf2memmap(f, root_path):

    memmap_map = dict()
    memmap_map['attributes'] = dict()
    memmap_map['attributes']['sorted_group_keys'] = f.get_node_attr('/', 'root_attribute').sorted_group_keys

    memmap_map['groups'] = dict()

    for group_key in memmap_map['attributes']['sorted_group_keys']:

        memmap_map['groups'][group_key] = dict()
        memmap_map['groups'][group_key]['attributes'] = dict()

        memmap_map['groups'][group_key]['attributes']['numeric_keys'] = \
            f.get_node_attr('/{}'.format(group_key), 'group_attribute').numeric_keys
        memmap_map['groups'][group_key]['attributes']['missing_keys'] = \
            f.get_node_attr('/{}'.format(group_key), 'group_attribute').missing_keys

        group_obj = f.get_node('/{}'.format(group_key))
        memmap_map['groups'][group_key]['arrays'] = dict()

        for key, array_obj in group_obj._v_leaves.items():

            memmap_map['groups'][group_key]['arrays'][key] = dict()

            array_dirpath = os.path.join(root_path, group_key)
            try:
                os.makedirs(array_dirpath)
            except:
                pass  # FIX THIS ASAP

            filepath = os.path.join(array_dirpath, key)
            array = array_obj[:]
            dtype = str(array.dtype)
            shape = array.shape

            memmap_map['groups'][group_key]['arrays'][key]['filepath'] = filepath
            memmap_map['groups'][group_key]['arrays'][key]['dtype'] = dtype
            memmap_map['groups'][group_key]['arrays'][key]['shape'] = shape

            write_memmap(filepath, dtype, shape, array)

    return memmap_map

@failed_method_retry
def write_memmap(filepath, dtype, shape, array):
    writable_memmap = np.memmap(filepath, dtype=dtype, mode="w+", shape=shape)
    writable_memmap[:] = array[:]
    del writable_memmap

@failed_method_retry
def read_memmap(filepath, dtype, shape, idx=None):
    readonly_memmap = np.memmap(filepath, dtype=dtype, mode="r", shape=shape)

    if idx is None:
    	array = readonly_memmap[:]
    else:
    	array = readonly_memmap[idx]

    del readonly_memmap
    return array


def save_obj(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

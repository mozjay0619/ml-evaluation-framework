from evaluation_framework.utils.pandas_utils import cast_datetime2int64
from evaluation_framework.utils.pandas_utils import cast_int64_2datetime
from evaluation_framework.utils.pandas_utils import encode_str2bytes
from evaluation_framework.utils.pandas_utils import encode_date_sequence
from evaluation_framework.utils.s3_utils import s3_upload_object
from evaluation_framework.utils.s3_utils import s3_download_object
from evaluation_framework.utils.s3_utils import s3_upload_zip_dir
from evaluation_framework.utils.s3_utils import s3_delete_object
from evaluation_framework.utils.zip_utils import unzip_dir
from evaluation_framework.utils.objectIO_utils import save_obj
from evaluation_framework.utils.objectIO_utils import load_obj
from evaluation_framework.utils.memmap_utils import write_memmap
from evaluation_framework.utils.memmap_utils import read_memmap
from evaluation_framework.utils.decorator_utils import failed_method_retry

import os
import shutil
from collections import namedtuple
import pickle
import numpy as np


@failed_method_retry
def load_local_data(evaluation_manager):

    memmap_root_dirpath = os.path.join(os.getcwd(), evaluation_manager.memmap_root_dirname)

    try:
        os.makedirs(memmap_root_dirpath)
    except:
        shutil.rmtree(memmap_root_dirpath)
        os.makedirs(memmap_root_dirpath)

    if evaluation_manager.return_predictions:

        prediction_records_dirpath = os.path.join(os.getcwd(), evaluation_manager.prediction_records_dirname)

        try:
            os.makedirs(prediction_records_dirpath)
        except:
            shutil.rmtree(prediction_records_dirpath)
            os.makedirs(prediction_records_dirpath)

    memmap_map = _write_memmap_filesys(evaluation_manager, memmap_root_dirpath)
    return memmap_map

def upload_local_data(task_manager):

    memmap_root_dirpath = os.path.join(os.getcwd(), task_manager.memmap_root_dirpath)
    s3_url = task_manager.S3_path
    object_name = task_manager.memmap_root_S3_object_name + '.zip'
    s3_upload_zip_dir(memmap_root_dirpath, s3_url, object_name)

def download_local_data(task_manager):
    """
    1. create memmap dir
    3. translate hdf5 to memmap
    4. graph will just use the memmap dirname to read it off from the "current pos"

    """
    s3_download_object(os.getcwd(), task_manager.S3_path, task_manager.memmap_root_S3_object_name + '.zip')

    zipped_filepath = os.path.join(os.getcwd(), task_manager.memmap_root_S3_object_name + '.zip')
    unzip_dir(zipped_filepath, task_manager.memmap_root_dirname)

    # update memmap_map with new root_dir
    updated_memmap_map_root_dirpath = os.path.join(os.getcwd(), task_manager.memmap_root_dirname)
    memmap_map_filepath = os.path.join(updated_memmap_map_root_dirpath, 'memmap_map')
    memmap_map = load_obj(memmap_map_filepath)
    memmap_map['root_dirpath'] = updated_memmap_map_root_dirpath
    save_obj(memmap_map, memmap_map_filepath)

    if task_manager.return_predictions:

        prediction_records_dirpath = os.path.join(os.getcwd(), task_manager.prediction_records_dirname)

        try:
            os.makedirs(prediction_records_dirpath)
        except:
            shutil.rmtree(prediction_records_dirpath)
            os.makedirs(prediction_records_dirpath)

def upload_remote_data(task_manager, ip_addr):
    """
    1. zip the prediction array directory
    2. send them to S3 bucket

    **The file structure should be identical to that of local machine, making it possible to
    use this method on local machine as well for testing purposes.
    """
    source_dirpath = os.path.join(os.getcwd(), 'prediction_arrays')
    
    job_uuid = task_manager.memmap_root_S3_object_name.split('__')[-1]
    host_uuid = ip_addr.replace('.', '-')
    object_name = 'prediction_arrays' + '__' + job_uuid + '__' + host_uuid + '.zip'
    
    s3_url = task_manager.S3_path
    s3_upload_zip_dir(source_dirpath, s3_url, object_name)
    
    return object_name

def download_remote_data(task_manager):
    """
    1. download the prediction array zip dirs from S3
    2. unzip them and place them into the same directory
    """
    job_uuid = task_manager.memmap_root_S3_object_name.split('__')[-1]
    prefix_name = 'prediction_arrays' + '__' + job_uuid
    s3_download_object(os.getcwd(), task_manager.S3_path, prefix_name)
    tmp = os.listdir(os.getcwd())
    prediction_arrays_zips = [elem for elem in tmp if elem.startswith(prefix_name) & elem.endswith('zip')]
    
    for prediction_arrays_zip in prediction_arrays_zips:
        
        zipped_filepath = os.path.join(os.getcwd(), prediction_arrays_zip)
        unzip_dir(zipped_filepath, os.path.join(os.getcwd(), 'prediction_arrays'))
        
def _write_memmap_filesys(task_manager, root_dirpath):
    """memmap mimicking hdf5 filesystem. 
    root_dirpath/
        memmap_map
        groupA__groupA'__arrayA (array)
        groupA__groupA'__arrayB (array)  
        ... etc


    root_dirpath / group_dirpath / filepath
    memmap['groups'][group_key]['groups'][group_key_innder]['arrays'][filepath, dtype, shape]

    """
    memmap_map = dict()
    memmap_map['attributes'] = dict()
    memmap_map['groups'] = dict()  # i.e. we define the first level to be groups
    memmap_map['root_dirpath'] = root_dirpath

    memmap_map_filepath = os.path.join(root_dirpath, 'memmap_map')

    group_key_size_tuples = []

    for group_key, grouped_pdf in task_manager.data.groupby(by=task_manager.groupby):

        group_key_size_tuples.append([group_key, len(grouped_pdf)])

        if task_manager.orderby:
            grouped_pdf = grouped_pdf.sort_values(by=task_manager.orderby)
        grouped_pdf = grouped_pdf.reset_index(drop=True)

        memmap_map['groups'][group_key] = dict()
        source_dirpath = memmap_map['root_dirpath']
        # memmap_map['groups'][group_key]['group_dirpath'] = '__'.join((source_dirpath, group_key))

        # memmap_map['groups'][group_key]['group_dirpath'] = os.path.join(source_dirpath, group_key)  
        memmap_map['groups'][group_key]['group_dirpath'] = group_key

        # NOTE FOR HMF: the first level after root_dirpath needs to be os path join not '__'
        # NOTE FOR HMF: need a check for this for generalized self-describing memmap tool
        # NOTE FOR HMF FROM DUAL CLIENT PERSPECTIVE: we want to leave the root_dirpath flexible
        # so for the first group level, we need to start a new path so we can join with an updated root_dirpath later
        memmap_map['groups'][group_key]['attributes'] = dict()
        memmap_map['groups'][group_key]['arrays'] = dict()  # i.e. we define the second level to be arrays
        
        # NOTE FOR HMF: later, develop this into recursive function and a generalized open source tool

        _write_datetime_types(task_manager, memmap_map, group_key, grouped_pdf)
        _write_str_types(task_manager, memmap_map, group_key, grouped_pdf)
        _write_numeric_types(task_manager, memmap_map, group_key, grouped_pdf)

        if task_manager.orderby:
            _write_orderby_array(task_manager, memmap_map, group_key, grouped_pdf)

        memmap_map['groups'][group_key]['attributes']['numeric_keys'] = task_manager.numeric_types
        memmap_map['groups'][group_key]['attributes']['missing_keys'] = task_manager.missing_keys

    group_key_size_tuples = sorted(group_key_size_tuples, key=lambda x: x[1])
    sorted_group_keys = [elem[0] for elem in group_key_size_tuples]
    memmap_map['attributes']['sorted_group_keys'] = sorted_group_keys

    save_obj(memmap_map, memmap_map_filepath)

    return memmap_map

def _write_datetime_types(task_manager, memmap_map, group_key, grouped_pdf):

    for key in task_manager.missing_keys['datetime_types']:

        # memmap array info
        array = cast_datetime2int64(grouped_pdf[key]).values
        dtype = str(array.dtype)
        shape = array.shape

        memmap_map['groups'][group_key]['arrays'][key] = dict()  # each array object is a dict as well for memmap case, unlike self documenting hdf5

        source_dirpath = memmap_map['groups'][group_key]['group_dirpath']
        filepath = '__'.join((source_dirpath, key))
        memmap_map['groups'][group_key]['arrays'][key]['filepath'] = filepath
        memmap_map['groups'][group_key]['arrays'][key]['dtype'] = dtype
        memmap_map['groups'][group_key]['arrays'][key]['shape'] = shape

        # later, develop this into recursive function and a generalized open source tool

        write_memmap(
            os.path.join(
                memmap_map['root_dirpath'], 
                memmap_map['groups'][group_key]['arrays'][key]['filepath']), 
            memmap_map['groups'][group_key]['arrays'][key]['dtype'], 
            memmap_map['groups'][group_key]['arrays'][key]['shape'], 
            array)

def _write_str_types(task_manager, memmap_map, group_key, grouped_pdf):

    for key in task_manager.missing_keys['str_types']:

        array = encode_str2bytes(grouped_pdf[key])
        dtype = str(array.dtype)
        shape = array.shape

        memmap_map['groups'][group_key]['arrays'][key] = dict()  # each array object is a dict as well for memmap case, unlike self documenting hdf5

        source_dirpath = memmap_map['groups'][group_key]['group_dirpath']
        filepath = '__'.join((source_dirpath, key))
        memmap_map['groups'][group_key]['arrays'][key]['filepath'] = filepath
        memmap_map['groups'][group_key]['arrays'][key]['dtype'] = dtype
        memmap_map['groups'][group_key]['arrays'][key]['shape'] = shape

        write_memmap(
            os.path.join(
                memmap_map['root_dirpath'], 
                memmap_map['groups'][group_key]['arrays'][key]['filepath']), 
            memmap_map['groups'][group_key]['arrays'][key]['dtype'], 
            memmap_map['groups'][group_key]['arrays'][key]['shape'], 
            array)

def _write_numeric_types(task_manager, memmap_map, group_key, grouped_pdf):

    key = 'numeric_types'

    # the code below can be made uniform
    # the below part corresponds to pytable's create_array(group_object, key, array) method
    # can be made into a generalized tool

    array = grouped_pdf[task_manager.numeric_types].values
    dtype = str(array.dtype)
    shape = array.shape

    memmap_map['groups'][group_key]['arrays'][key] = dict()

    source_dirpath = memmap_map['groups'][group_key]['group_dirpath']
    filepath = '__'.join((source_dirpath, key))
    memmap_map['groups'][group_key]['arrays'][key]['filepath'] = filepath
    memmap_map['groups'][group_key]['arrays'][key]['dtype'] = dtype
    memmap_map['groups'][group_key]['arrays'][key]['shape'] = shape

    write_memmap(
        os.path.join(
            memmap_map['root_dirpath'], 
            memmap_map['groups'][group_key]['arrays'][key]['filepath']), 
        memmap_map['groups'][group_key]['arrays'][key]['dtype'], 
        memmap_map['groups'][group_key]['arrays'][key]['shape'], 
        array)

def _write_orderby_array(task_manager, memmap_map, group_key, grouped_pdf):

    key = 'orderby_array'

    array = encode_date_sequence(grouped_pdf[task_manager.orderby]).values
    dtype = str(array.dtype)
    shape = array.shape

    memmap_map['groups'][group_key]['arrays'][key] = dict()

    source_dirpath = memmap_map['groups'][group_key]['group_dirpath']
    filepath = '__'.join((source_dirpath, key))
    memmap_map['groups'][group_key]['arrays'][key]['filepath'] = filepath
    memmap_map['groups'][group_key]['arrays'][key]['dtype'] = dtype
    memmap_map['groups'][group_key]['arrays'][key]['shape'] = shape

    write_memmap(
        os.path.join(
            memmap_map['root_dirpath'], 
            memmap_map['groups'][group_key]['arrays'][key]['filepath']), 
        memmap_map['groups'][group_key]['arrays'][key]['dtype'], 
        memmap_map['groups'][group_key]['arrays'][key]['shape'], 
        array)


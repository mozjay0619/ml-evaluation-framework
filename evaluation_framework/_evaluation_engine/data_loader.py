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
from evaluation_framework import constants

import HMF

import os
import shutil
from collections import namedtuple
import pickle
import numpy as np
import time


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
    memmap_map_filepath = os.path.join(updated_memmap_map_root_dirpath, constants.EF_MEMMAP_MAP_NAME)
    memmap_map = load_obj(memmap_map_filepath)
    memmap_map['dirpath'] = updated_memmap_map_root_dirpath
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

    **ip_addr is added at the decorator [ yarn_directory_normalizer ]
    """
    source_dirpath = os.path.join(os.getcwd(), 'prediction_arrays')
    
    host_uuid = ip_addr.replace('.', '-')
    object_name = 'prediction_arrays' + '__' + task_manager.job_uuid + '__' + host_uuid + '.zip'
    
    s3_url = task_manager.S3_path
    s3_upload_zip_dir(source_dirpath, s3_url, object_name)
    
    return object_name

def download_remote_data(task_manager):
    """
    1. download the prediction array zip dirs from S3
    2. unzip them and place them into the same directory
    """

    prediction_dirpath = os.path.join(task_manager.evaluation_task_dirpath, task_manager.prediction_records_dirname)
    prediction_filenames = os.listdir(prediction_dirpath)

    prefix_name = 'prediction_arrays' + '__' + task_manager.job_uuid
    s3_download_object(prediction_dirpath, task_manager.S3_path, prefix_name)
    prediction_filenames = os.listdir(prediction_dirpath)
    prediction_arrays_zips = [elem for elem in prediction_filenames if elem.startswith(prefix_name) & elem.endswith('zip')]
    
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

    f = HMF.open_file(root_dirpath, mode='w+')

    f.from_pandas(task_manager.data, groupby=task_manager.groupby, orderby=task_manager.orderby)

    f.register_array('numeric_types', task_manager.numeric_types)
    f.register_array('orderby_array', constants.EF_ORDERBY_NAME)

    for i in range(len(f.group_names)):

        f.set_node_attr('/{}'.format(f.group_names[i]), key='numeric_keys', value=task_manager.numeric_types)
        f.set_node_attr('/{}'.format(f.group_names[i]), key='missing_keys', value=task_manager.missing_keys)

    group_key_size_tuples = sorted(zip(f.group_names, f.group_sizes), key=lambda x: x[1])
    sorted_group_keys = [elem[0] for elem in group_key_size_tuples]
    f.set_node_attr('/', key='sorted_group_keys', value=sorted_group_keys)

    f.close()


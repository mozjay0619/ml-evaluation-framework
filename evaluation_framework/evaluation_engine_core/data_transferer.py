


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
    memmap_map_filepath = os.path.join(updated_memmap_map_root_dirpath, constants.HMF_MEMMAP_MAP_NAME)
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
        
        zipped_filepath = os.path.join(prediction_dirpath, prediction_arrays_zip)
        unzip_dir(zipped_filepath, prediction_dirpath)
        
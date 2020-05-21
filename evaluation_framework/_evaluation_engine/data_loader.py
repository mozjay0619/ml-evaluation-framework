from evaluation_framework.utils.pandas_utils import cast_datetime2int64
from evaluation_framework.utils.pandas_utils import cast_int64_2datetime
from evaluation_framework.utils.pandas_utils import encode_str2bytes
from evaluation_framework.utils.pandas_utils import encode_date_sequence

from evaluation_framework.utils.s3_utils import s3_upload_object
from evaluation_framework.utils.s3_utils import s3_download_object
from evaluation_framework.utils.s3_utils import s3_upload_zip_dir
from evaluation_framework.utils.s3_utils import s3_delete_object

from evaluation_framework.utils.zip_utils import unzip_dir

from .memmap_layer import translate_hdf2memmap
from .memmap_layer import save_obj


import os
import shutil
import tables
from collections import namedtuple
import pickle


RootAttribute = namedtuple('RootAttribute', [
    'sorted_group_keys'])

GroupAttribute = namedtuple('GroupAttribute', [
    'numeric_keys',
    'missing_keys'])


class DataLoader():
    """straight forward procedures here"""
    
    def __init__(self, evaluation_manager=None):
        
        # if S3_path = None, don't save to S3
        # self.evaluation_manager = evaluation_manager
        # self.local_dirpath = evaluation_manager.local_directory_path
        # self.S3_path = evaluation_manager.S3_path
        # self.overwrite = True  # remove and make it datetime indexed dir!
        
        # if self.local_dirpath:
        #     self.check_local_dirpath()
        self.evaluation_manager = evaluation_manager


    def load_local_data(self):

        hdf5_filepath = os.path.join(os.getcwd(), self.evaluation_manager.hdf5_filename)
        memmap_root_dirpath = os.path.join(os.getcwd(), self.evaluation_manager.memmap_root_dirname)
        os.makedirs(memmap_root_dirpath)

        self.open_hdf5_file()
        self.write_hdf5_file()

        memmap_map = translate_hdf2memmap(self.hdf5_fileobj, self.evaluation_manager.memmap_root_dirpath)
        memmap_map_filepath = os.path.join(self.evaluation_manager.memmap_root_dirpath, 'memmap_map')
        save_obj(memmap_map, memmap_map_filepath)

        self.close_hdf5_file()

    def save_to_s3(self):

        memmap_root_dirpath = os.path.join(os.getcwd(), self.evaluation_manager.memmap_root_dirname)
        s3_url = self.evaluation_manager.S3_path
        object_name = self.evaluation_manager.memmap_root_dirname + '.zip'
        s3_upload_zip_dir(memmap_root_dirpath, s3_url, object_name)


    def load_remote_data(self):
        """
        1. create memmap dir
        3. translate hdf5 to memmap
        4. graph will just use the memmap dirname to read it off from the "current pos"

        """

        s3_download_object(os.getcwd(), self.evaluation_manager.S3_path, self.evaluation_manager.memmap_root_dirname + '.zip')

        zipped_filepath = os.path.join(os.getcwd(), self.evaluation_manager.memmap_root_dirname + '.zip')
        unzip_dir(zipped_filepath, self.evaluation_manager.memmap_root_dirname)

        
    def load_data(self):
        
        
        self.local_dirpath = self.evaluation_manager.local_directory_path
        self.S3_path = self.evaluation_manager.S3_path or None
        self.overwrite = True
        
        # self.check_local_dirpath()
#         self.save_feather()
        self.open_hdf5_file()
        self.write_hdf5_file()

        memmap_map = translate_hdf2memmap(self.hdf5_fileobj, self.evaluation_manager.memmap_root_dirpath)
        memmap_map_filepath = os.path.join(self.evaluation_manager.memmap_root_dirpath, 'memmap_map')
        save_obj(memmap_map, memmap_map_filepath)

        self.close_hdf5_file()


        
    # def check_local_dirpath(self):
        
    #     if os.path.exists(self.local_dirpath) and not self.overwrite:
    #         raise ValueError('[ local_dirpath ] already exists. Set [ overwrite ] flag to True '
    #                          'to use this directory path.')
            
    #     if os.path.exists(self.local_dirpath):
    #         shutil.rmtree(self.local_dirpath)
            
    #     os.mkdir(self.local_dirpath)
    
    def open_hdf5_file(self):
        
        hdf5_filepath = os.path.join(os.getcwd(), self.evaluation_manager.hdf5_filename )
        self.hdf5_fileobj = tables.open_file(hdf5_filepath, 'w')    
        
    def write_hdf5_file(self):

        group_key_size_tuples = []
        
        
        for group_key, grouped_pdf in self.evaluation_manager.data.groupby(by=self.evaluation_manager.groupby):

            # data_array_names = dict()

            group_key_size_tuples.append([group_key, len(grouped_pdf)])
    
            group_obj = self.hdf5_fileobj.create_group("/", group_key)
        
            grouped_pdf = grouped_pdf.reset_index(drop=True)
            
            self._load_datetime_types(group_obj, grouped_pdf)
            self._load_str_types(group_obj, grouped_pdf)
            self._load_numeric_types(group_obj, grouped_pdf)

            if self.evaluation_manager.orderby:

                self._load_orderby_array(group_obj, grouped_pdf)

            group_attr = GroupAttribute(
                numeric_keys=self.evaluation_manager.numeric_types,
                missing_keys=self.evaluation_manager.missing_keys)
            self.hdf5_fileobj.set_node_attr('/{}'.format(group_key), 'group_attribute', group_attr)

        group_key_size_tuples = sorted(group_key_size_tuples, key=lambda x: x[1])
        sorted_group_keys = [elem[0] for elem in group_key_size_tuples]
        root_attr = RootAttribute(sorted_group_keys=sorted_group_keys)
        self.hdf5_fileobj.set_node_attr('/', 'root_attribute', root_attr)



    # def _load_group_uuid_array(self, hdf5_fileobj, group_obj, evaluation_manager, grouped_pdf, data_array_names):

    #     uuid_array = np.arange(len(grouped_pdf))


    
    def _load_datetime_types(self, group_obj, grouped_pdf):

        for key in self.evaluation_manager.missing_keys['datetime_types']:

            tmp_array = cast_datetime2int64(grouped_pdf[key]).values
            self.hdf5_fileobj.create_array(group_obj, key, tmp_array)



    def _load_str_types(self, group_obj, grouped_pdf):

        for key in self.evaluation_manager.missing_keys['str_types']:

            tmp_array = encode_str2bytes(grouped_pdf[key])
            self.hdf5_fileobj.create_array(group_obj, key, tmp_array)

    # target separate out later!

    def _load_numeric_types(self, group_obj, grouped_pdf):

        self.hdf5_fileobj.create_array(
        	group_obj, 
        	'numeric_types', 
        	grouped_pdf[self.evaluation_manager.numeric_types].values)

    def _load_orderby_array(self, group_obj, grouped_pdf):

        self.hdf5_fileobj.create_array(
            group_obj, 
            'orderby_array',
            encode_date_sequence(grouped_pdf[self.evaluation_manager.orderby]).values)



        
    def close_hdf5_file(self):
        
        self.hdf5_fileobj.close()
        
    
    def save_feather(self):
        # may not need to! if em data is not altered
        pass
    
    def load_feather(self):
        pass
    
    def save_to_S3(self):
        pass
    
    def load_from_S3(self):
        pass
    
#     def 
        
        
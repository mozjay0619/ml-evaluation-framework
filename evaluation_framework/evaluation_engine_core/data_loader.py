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



import HMF
import numpy as np
import pandas as pd
import copy

class DataLoader():
    """
    This class holds the HMF object. 
    """
    
    def __init__(self, dirpath, overwrite=False):
        
        if overwrite:
            mode = 'w+'
        else:
            mode = 'r+'
                    
        self.f = HMF.open_file(dirpath, mode=mode)
        self.dirpath = dirpath
        
#     def create_dirpaths(self, memmap_root_dirname, return_predictions, prediction_records_dirname=None):
        
#         memmap_root_dirpath = os.path.join(os.getcwd(), memmap_root_dirname)
#         try:
#             os.makedirs(memmap_root_dirpath)
#         except:
#             shutil.rmtree(memmap_root_dirpath)
#             os.makedirs(memmap_root_dirpath)
            
#         if return_predictions:

#             prediction_records_dirpath = os.path.join(os.getcwd(), prediction_records_dirname)
#             try:
#                 os.makedirs(prediction_records_dirpath)
#             except:
#                 shutil.rmtree(prediction_records_dirpath)
#                 os.makedirs(prediction_records_dirpath)

    def save_data(self, pdf, orderby, groupby, numeric_columns, missing_keys):
        """memmap mimicking hdf5 filesystem. 
        root_dirpath/
            memmap_map
            groupA__groupA'__arrayA (array)
            groupA__groupA'__arrayB (array)  
            ... etc


        root_dirpath / group_dirpath / filepath
        memmap['groups'][group_key]['groups'][group_key_innder]['arrays'][filepath, dtype, shape]

        """
        self.f.from_pandas(pdf, groupby=groupby, orderby=orderby)
        self.f.register_array('numeric_types', numeric_columns)
        self.f.register_array('orderby_array', constants.EF_ORDERBY_NAME)
        
        for i in range(len(self.f.get_group_names())):

            self.f.set_node_attr('/{}'.format(self.f.get_group_names()[i]), 
                                 key='numeric_keys', value=numeric_columns)
            self.f.set_node_attr('/{}'.format(self.f.get_group_names()[i]), 
                                 key='missing_keys', value=missing_keys)
        
        # group_key_size_tuples = sorted(zip(self.f.get_group_names(), self.f.group_sizes), 
        #                                key=lambda x: x[1], reverse=True)
        # sorted_group_keys = [elem[0] for elem in group_key_size_tuples]
        # self.f.set_node_attr('/', key='sorted_group_keys', value=sorted_group_keys)
        
        self.f.close()
        
    def load_data(self, group_key, data_idx):
        
        missing_keys = self.f.get_node_attr('/{}'.format(group_key), key='missing_keys')
        data_colnames = copy.copy(self.f.get_node_attr('/{}'.format(group_key), key='numeric_keys'))
        data_arrays = [self.f.get_array('/{}/numeric_types'.format(group_key), idx=data_idx)]
        
        for colname in missing_keys['datetime_types']:
            tmp_array = self.f.get_array('/{}/{}'.format(group_key, colname))
            data_arrays.append(tmp_array.reshape(-1, 1))
            data_colnames.append(colname)
            
        data_array = np.hstack(data_arrays)
        pdf = pd.DataFrame(data_array, columns=data_colnames)
        
        for i in range(len(missing_keys['datetime_types'])):
            pdf.iloc[:, i-1] = pd.to_datetime(pdf.iloc[:, i-1])
            
        for colname in missing_keys['str_types']:
            tmp_array = self.f.get_array('/{}/{}'.format(group_key, colname))
            tmp_array = tmp_array.astype(str)
            pdf[colname] = tmp_array
            
        return pdf






# # from ..utils.data_structure_utils import return_indices

# import HMF
# import numpy as np
# import pandas as pd

# class DataLoader():
    
#     def init(self, dirpath):
        
#         self.f = HMF.open_file(dirpath, mode='r+')
#         self.dirpath = dirpath

    
#     @staticmethod
#     def save_data(dirpath, data, orderby=None, groupby=None):
#         """Later, add support for groupby
        
#         """
        
#         f = HMF.open_file(dirpath, mode='w+')
#         numeric_data = data.select_dtypes(include=[np.number])
        
#         f.from_pandas(numeric_data, orderby=orderby, groupby=groupby)

#         f.register_array('data_array', numeric_data.columns)
#         f.set_node_attr('/column_names', key='column_names', value=numeric_data.columns)

#         f.close()
        

#     def load_data(self, features, target=None, group_key=None):

        
#         if not group_key:
            
#             data_array = self.f.get_array('/data_array')
#             column_names = list(self.f.get_node_attr('/column_names', key='column_names'))
        
#         else:

#             data_array = self.f.get_array('/{}/data_array'.format(group_key))
#             column_names = self.f.get_node_attr('/{}/column_names'.format(group_key), key='column_names')

        
#         feature_column_indices = return_indices(column_names, features)
#         X = data_array[:, feature_column_indices]

#         if target:

#             target_column_index = return_indices(column_names, [target])
#             y = data_array[:, target_column_index]

#             return X, y

#         else:

#             return X


#     def load_dataframe(self, features, group_key=None):

        
#         if not group_key:
#             data_array = self.f.get_array('/data_array')
#             column_names = list(self.f.get_node_attr('/column_names', key='column_names'))

#         else:

#             data_array = self.f.get_array('/{}/data_array'.format(group_key))
#             column_names = self.f.get_node_attr('/{}/column_names'.format(group_key), key='column_names')


#         return pd.DataFrame(data_array, columns=column_names)

#     def load_feature_stack(self, group_key=None):

#         return self.f.get_node_attr('/{}/all_corr_features'.format(group_key), key='all_corr_features')












# @failed_method_retry
# def load_local_data(evaluation_manager):

#     memmap_root_dirpath = os.path.join(os.getcwd(), evaluation_manager.memmap_root_dirname)

#     try:
#         os.makedirs(memmap_root_dirpath)
#     except:
#         shutil.rmtree(memmap_root_dirpath)
#         os.makedirs(memmap_root_dirpath)

#     if evaluation_manager.return_predictions:

#         prediction_records_dirpath = os.path.join(os.getcwd(), evaluation_manager.prediction_records_dirname)

#         try:
#             os.makedirs(prediction_records_dirpath)
#         except:
#             shutil.rmtree(prediction_records_dirpath)
#             os.makedirs(prediction_records_dirpath)

#     memmap_map = _write_memmap_filesys(evaluation_manager, memmap_root_dirpath)
#     return memmap_map


#     def _write_memmap_filesys(task_manager, root_dirpath):
#         """memmap mimicking hdf5 filesystem. 
#         root_dirpath/
#             memmap_map
#             groupA__groupA'__arrayA (array)
#             groupA__groupA'__arrayB (array)  
#             ... etc


#         root_dirpath / group_dirpath / filepath
#         memmap['groups'][group_key]['groups'][group_key_innder]['arrays'][filepath, dtype, shape]

#         """

#         f = HMF.open_file(root_dirpath, mode='w+')

#         f.from_pandas(task_manager.data, groupby=task_manager.groupby, orderby=task_manager.orderby)

#         f.register_array('numeric_types', task_manager.numeric_types)
#         f.register_array('orderby_array', constants.EF_ORDERBY_NAME)

#         for i in range(len(f.group_names)):

#             f.set_node_attr('/{}'.format(f.group_names[i]), key='numeric_keys', value=task_manager.numeric_types)
#             f.set_node_attr('/{}'.format(f.group_names[i]), key='missing_keys', value=task_manager.missing_keys)

#         group_key_size_tuples = sorted(zip(f.group_names, f.group_sizes), key=lambda x: x[1], reverse=True)
#         sorted_group_keys = [elem[0] for elem in group_key_size_tuples]
#         f.set_node_attr('/', key='sorted_group_keys', value=sorted_group_keys)

#         f.close()


from .memmap_layer import load_obj
from .memmap_layer import read_memmap

import tables
import copy
import numpy as np
import pandas as pd
import os

class TaskGraph():
    
    def __init__(self, evaluation_manager, cv):  # thinned out
        pass
        # no data in the em!
        
#      self.evaluation_manager = evaluation_manager
        # get cv object here
        self.evaluation_manager = evaluation_manager
        self.cv = cv

    def run(self, group_key, cv_split_index):   
        
        train_data, test_data = self.get_data(group_key, cv_split_index)
        evaluation_result = self.task_graph(train_data, test_data, group_key)
        return (group_key, cv_split_index, evaluation_result)

    def get_data(self, group_key, cv_split_index):

        memmap_root_dirpath = os.path.join(os.getcwd(), self.evaluation_manager.memmap_root_dirname)
        memmap_map_filepath = os.path.join(memmap_root_dirpath, 'memmap_map')
        memmap_map = load_obj(memmap_map_filepath)
        
        train_idx, test_idx = self._get_cross_validation_fold_idx(memmap_map, group_key, cv_split_index)

        train_data = self._read_memmap(memmap_map, group_key, train_idx)
        test_data = self._read_memmap(memmap_map, group_key, test_idx)

        return train_data, test_data

    def task_graph(self, train_data, test_data, group_key):  # groupkey is redundant info get rid of it
        
        configs = self.evaluation_manager.user_configs
        
        preprocessed_train_data = self.evaluation_manager.preprocess_train_data(
            train_data, 
            configs)
        
        trained_estimator = self.evaluation_manager.model_fit(
           preprocessed_train_data, 
           self.evaluation_manager.hyperparameters, 
           self.evaluation_manager.estimator,
           self.evaluation_manager.feature_names[group_key],
           self.evaluation_manager.target_name)

        preprocessed_test_data = self.evaluation_manager.preprocess_test_data(
           test_data, 
           preprocessed_train_data, 
           configs)

        prediction_result = self.evaluation_manager.model_predict(
           preprocessed_test_data, 
           trained_estimator, 
           self.evaluation_manager.feature_names[group_key],
           self.evaluation_manager.target_name)

        evaluation_result = self.evaluation_manager.evaluate_prediction(
           preprocessed_test_data, 
           prediction_result)

        return evaluation_result
        
    
  
    def _read_memmap(self, memmap_map, group_key, data_idx):
    
        missing_keys = memmap_map['groups'][group_key]['attributes']['missing_keys']
        data_colnames = copy.copy(memmap_map['groups'][group_key]['attributes']['numeric_keys']) 

        filepath = memmap_map['groups'][group_key]['arrays']['numeric_types']['filepath']
        dtype = memmap_map['groups'][group_key]['arrays']['numeric_types']['dtype']
        shape = memmap_map['groups'][group_key]['arrays']['numeric_types']['shape']
        data_arrays = [read_memmap(filepath, dtype, shape, data_idx)]

        for colname in missing_keys['datetime_types']:
        
            filepath = memmap_map['groups'][group_key]['arrays'][colname]['filepath']
            dtype = memmap_map['groups'][group_key]['arrays'][colname]['dtype']
            shape = memmap_map['groups'][group_key]['arrays'][colname]['shape']
            tmp_array = read_memmap(filepath, dtype, shape, data_idx)

            data_arrays.append(tmp_array.reshape(-1, 1))
            data_colnames.append(colname)
            
        data_array = np.hstack(data_arrays)
        pdf = pd.DataFrame(data_array, columns=data_colnames)
        
        for i in range(len(missing_keys['datetime_types'])):
            pdf.iloc[:, i-1] = pd.to_datetime(pdf.iloc[:, i-1])
            
        for colname in missing_keys['str_types']:
            filepath = memmap_map['groups'][group_key]['arrays'][colname]['filepath']
            dtype = memmap_map['groups'][group_key]['arrays'][colname]['dtype']
            shape = memmap_map['groups'][group_key]['arrays'][colname]['shape']
            tmp_array = read_memmap(filepath, dtype, shape, data_idx)

            tmp_array = tmp_array.astype(str)
            pdf[colname] = tmp_array

        return pdf
    
    def _get_cross_validation_fold_idx(self, memmap_map, group_key, cv_split_index):
        
        if self.evaluation_manager.orderby:  # have another parameter to check orderby needs to happen...
            # by cv scheme itself!
            
            # need to add random state

            filepath = memmap_map['groups'][group_key]['arrays']['orderby_array']['filepath']
            dtype = memmap_map['groups'][group_key]['arrays']['orderby_array']['dtype']
            shape = memmap_map['groups'][group_key]['arrays']['orderby_array']['shape']
            group_ordered_array = read_memmap(filepath, dtype, shape)

            for idx, (train, test) in enumerate(self.cv.split(group_ordered_array)):
                if idx == cv_split_index:
                    break
        
        return train, test
    
    
    




        
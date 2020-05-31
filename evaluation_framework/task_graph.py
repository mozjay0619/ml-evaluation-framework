from evaluation_framework.utils.objectIO_utils import load_obj
from evaluation_framework.utils.memmap_utils import write_memmap
from evaluation_framework.utils.memmap_utils import read_memmap

import copy
import numpy as np
import pandas as pd
import os


class TaskGraph():
    """
    The atomic task graph that is run by each dask process. 
    """
    
    def __init__(self, task_manager, cv): 

        self.task_manager = task_manager
        self.cv = cv

    def run(self, group_key, cv_split_index):   

        attempts = 0

        while attempts < 3:

            try:
                
                train_data, test_data, train_idx, test_idx = self.get_data(group_key, cv_split_index)
                prediction_result, evaluation_result = self.task_graph(train_data, test_data, group_key)

                if self.task_manager.return_predictions:
                    self.record_predictions(group_key, cv_split_index, prediction_result, test_data, test_idx)

                break

            except:

                attempts += 1

        train_data, test_data, train_idx, test_idx = self.get_data(group_key, cv_split_index)
        prediction_result, evaluation_result = self.task_graph(train_data, test_data, group_key)

        if self.task_manager.return_predictions:
            self.record_predictions(group_key, cv_split_index, prediction_result, test_data, test_idx)

        return (group_key, cv_split_index, evaluation_result, len(prediction_result))

    def get_data(self, group_key, cv_split_index):

        memmap_root_dirpath = os.path.join(os.getcwd(), self.task_manager.memmap_root_dirname)
        memmap_map_filepath = os.path.join(memmap_root_dirpath, 'memmap_map')
        self.memmap_map = load_obj(memmap_map_filepath)
        
        train_idx, test_idx = self._get_cross_validation_fold_idx(self.memmap_map, group_key, cv_split_index)

        train_data = self._read_memmap(self.memmap_map, group_key, train_idx)
        test_data = self._read_memmap(self.memmap_map, group_key, test_idx)

        return train_data, test_data, train_idx, test_idx

    def task_graph(self, train_data, test_data, group_key):  # groupkey is redundant info get rid of it
        
        configs = self.task_manager.user_configs
        
        preprocessed_train_data = self.task_manager.preprocess_train_data(
            train_data, 
            configs)
        
        trained_estimator = self.task_manager.model_fit(
           preprocessed_train_data, 
           self.task_manager.hyperparameters, 
           self.task_manager.estimator,
           self.task_manager.feature_names[group_key],
           self.task_manager.target_name)

        preprocessed_test_data = self.task_manager.preprocess_test_data(
           test_data, 
           preprocessed_train_data, 
           configs)

        prediction_result = self.task_manager.model_predict(
           preprocessed_test_data, 
           trained_estimator, 
           self.task_manager.feature_names[group_key],
           self.task_manager.target_name)

        evaluation_result = self.task_manager.evaluate_prediction(
           preprocessed_test_data, 
           prediction_result['specialEF_float32_predictions'])

        return (prediction_result, evaluation_result)
        
    def _read_memmap(self, memmap_map, group_key, data_idx):
    
        missing_keys = memmap_map['groups'][group_key]['attributes']['missing_keys']
        data_colnames = copy.copy(memmap_map['groups'][group_key]['attributes']['numeric_keys']) 

        filepath = os.path.join(memmap_map['root_dirpath'], memmap_map['groups'][group_key]['arrays']['numeric_types']['filepath'])
        dtype = memmap_map['groups'][group_key]['arrays']['numeric_types']['dtype']
        shape = memmap_map['groups'][group_key]['arrays']['numeric_types']['shape']
        data_arrays = [read_memmap(filepath, dtype, shape, data_idx)]

        for colname in missing_keys['datetime_types']:
        
            filepath = os.path.join(memmap_map['root_dirpath'], memmap_map['groups'][group_key]['arrays'][colname]['filepath'])
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
            
            filepath = os.path.join(memmap_map['root_dirpath'], memmap_map['groups'][group_key]['arrays'][colname]['filepath'])
            dtype = memmap_map['groups'][group_key]['arrays'][colname]['dtype']
            shape = memmap_map['groups'][group_key]['arrays'][colname]['shape']
            tmp_array = read_memmap(filepath, dtype, shape, data_idx)

            tmp_array = tmp_array.astype(str)
            pdf[colname] = tmp_array

        return pdf
    
    def _get_cross_validation_fold_idx(self, memmap_map, group_key, cv_split_index):
        
        if self.task_manager.orderby:  # have another parameter to check orderby needs to happen...
            # by cv scheme itself!
            
            # need to add random state

            filepath = os.path.join(memmap_map['root_dirpath'], memmap_map['groups'][group_key]['arrays']['orderby_array']['filepath'])
            dtype = memmap_map['groups'][group_key]['arrays']['orderby_array']['dtype']
            shape = memmap_map['groups'][group_key]['arrays']['orderby_array']['shape']
            group_ordered_array = read_memmap(filepath, dtype, shape)

            for idx, (train, test) in enumerate(self.cv.split(group_ordered_array)):
                if idx == cv_split_index:
                    break
        
        return train, test
    
    def record_predictions(self, group_key, cv_split_index, prediction_result, test_data, test_idx):
        """memmap['groups'][group_key]['groups'][group_key_innder]['arrays'][filepath, dtype, shape]

        """
        test_data_prediction = test_data.merge(prediction_result, on='specialEF_float32_UUID', how='inner')

        predictions_array = test_data_prediction[['specialEF_float32_UUID', 'specialEF_float32_predictions']]
        predictions_array = predictions_array.values.astype(np.float32)

        filename = '__'.join((group_key, str(cv_split_index))) + '.npy'
        filepath = os.path.join(os.getcwd(), self.task_manager.prediction_records_dirname, filename)

        # try:
        np.save(filepath, predictions_array)
        np.load(filepath)
        # except:
        #     pass
            # need to pass some value to indicate failure instead of unavailability!










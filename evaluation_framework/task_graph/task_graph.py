from evaluation_framework.utils.objectIO_utils import load_obj
from evaluation_framework.utils.memmap_utils import write_memmap
from evaluation_framework.utils.memmap_utils import read_memmap
from evaluation_framework import constants

import HMF

import copy
import numpy as np
import pandas as pd
import os
import time


class TaskGraph():
    """
    The atomic task graph that is run by each dask process. 
    
    Design philosophies:
    --------------------
    1. It should be independently runnable to validate the task.
    2. It should be independently designable only to be plugged into the Engine
    (designable up to the data directory structure)
    """
    
    def __init__(self, task_manager, cv, verbose=False): 

        self.task_manager = task_manager
        self.cv = cv
        self.verbose = verbose

        # root_dirpath = os.path.join(os.getcwd(), self.task_manager.memmap_root_dirname)
        # self.f = HMF.open_file(root_dirpath, mode='r+')

        # 

    def run(self, group_key, cv_split_index, data_loader):   

        # root_dirpath = os.path.join(os.getcwd(), self.task_manager.memmap_root_dirname)
        # self.f = HMF.open_file(root_dirpath, mode='r+')

        attempts = 0
        succeeded = False

        while attempts < 3:

            try:
                
                train_data, test_data, train_idx, test_idx, date_range = self.get_data(group_key, cv_split_index, data_loader)
                prediction_result, evaluation_result = self.task_graph(train_data, test_data, group_key)

                if self.task_manager.return_predictions:
                    self.record_predictions(group_key, cv_split_index, prediction_result, test_data, test_idx)

                succeeded = True

                break

            except:

                attempts += 1

        if not succeeded:

            train_data, test_data, train_idx, test_idx, date_range = self.get_data(group_key, cv_split_index, data_loader)
            prediction_result, evaluation_result = self.task_graph(train_data, test_data, group_key)

            if self.task_manager.return_predictions:
                self.record_predictions(group_key, cv_split_index, prediction_result, test_data, test_idx)

        return (group_key, cv_split_index, evaluation_result, self.train_data_size, self.test_data_size, list(date_range), self.task_duration)

    def get_data(self, group_key, cv_split_index, data_loader):



        # memmap_root_dirpath = os.path.join(os.getcwd(), self.task_manager.memmap_root_dirname)
        # memmap_map_filepath = os.path.join(memmap_root_dirpath, constants.HMF_MEMMAP_MAP_NAME + '0')
        # self.memmap_map = load_obj(memmap_map_filepath)
        # self.memmap_map = self.f.memmap_map

        memmap_map = data_loader.f.memmap_map

        
        train_idx, test_idx, date_range = self._get_cross_validation_fold_idx(memmap_map, group_key, cv_split_index, data_loader)

        if self.verbose:
            print('train size: {}'.format(len(train_idx)))
            print('test_size: {}'.format(len(test_idx)))

        train_data = self._read_memmap(memmap_map, group_key, train_idx, data_loader)
        test_data = self._read_memmap(memmap_map, group_key, test_idx, data_loader)

        return train_data, test_data, train_idx, test_idx, date_range

    def task_graph(self, train_data, test_data, group_key):  # groupkey is redundant info get rid of it

        task_start_time = time.time()
        
        configs = self.task_manager.user_configs
        
        if self.verbose: start_time = time.time()
        preprocessed_train_data = self.task_manager.preprocess_train_data(
            train_data, 
            configs)
        if self.verbose: print('Completed preprocess_train_data:', time.time() - start_time)

        self.train_data_size = len(preprocessed_train_data)

        if self.task_manager.hyperparameters is not None:
            hyperparameters = self.task_manager.hyperparameters[group_key]
        else:
            hyperparameters = None
        
        if self.verbose: start_time = time.time()
        trained_estimator = self.task_manager.model_fit(
           preprocessed_train_data, 
           hyperparameters, 
           self.task_manager.estimator,
           self.task_manager.feature_names[group_key],
           self.task_manager.target_name)
        if self.verbose: print('Completed model_fit:', time.time() - start_time)

        if self.verbose: start_time = time.time()
        preprocessed_test_data = self.task_manager.preprocess_test_data(
           test_data, 
           preprocessed_train_data, 
           configs)
        if self.verbose: print('Completed preprocess_test_data:', time.time() - start_time)

        if self.verbose: start_time = time.time()
        prediction_result = self.task_manager.model_predict(
           preprocessed_test_data, 
           trained_estimator, 
           self.task_manager.feature_names[group_key],
           self.task_manager.target_name)
        if self.verbose: print('Completed model_predict:', time.time() - start_time)

        self.test_data_size = len(prediction_result)

        if self.verbose: start_time = time.time()
        evaluation_result = self.task_manager.evaluate_prediction(
           preprocessed_test_data, 
           prediction_result[constants.EF_PREDICTION_NAME])
        if self.verbose: print('Completed evaluate_prediction:', time.time() - start_time)

        self.task_duration = time.time() - task_start_time

        return (prediction_result, evaluation_result)
        
    def _read_memmap(self, memmap_map, group_key, data_idx, data_loader):

        missing_keys = data_loader.f.get_node_attr('/{}'.format(group_key), key='missing_keys')
        data_colnames = copy.copy(data_loader.f.get_node_attr('/{}'.format(group_key), key='numeric_keys'))

        data_arrays = [data_loader.f.get_array('/{}/numeric_types'.format(group_key), idx=data_idx)]

        for colname in missing_keys['datetime_types']:

            tmp_array = data_loader.f.get_array('/{}/{}'.format(group_key, colname))

            data_arrays.append(tmp_array.reshape(-1, 1))
            data_colnames.append(colname)
            
        data_array = np.hstack(data_arrays)
        pdf = pd.DataFrame(data_array, columns=data_colnames)
        
        for i in range(len(missing_keys['datetime_types'])):
            pdf.iloc[:, i-1] = pd.to_datetime(pdf.iloc[:, i-1])
            
        for colname in missing_keys['str_types']:

            tmp_array = data_loader.f.get_array('/{}/{}'.format(group_key, colname))

            tmp_array = tmp_array.astype(str)
            pdf[colname] = tmp_array

        
        return pdf
    
    def _get_cross_validation_fold_idx(self, memmap_map, group_key, cv_split_index, data_loader):

        
        if self.task_manager.orderby:  # have another parameter to check orderby needs to happen...
            # by cv scheme itself!
            
            # need to add random state

            group_ordered_array = data_loader.f.get_array('/{}/orderby_array'.format(group_key))

            for idx, (train, test, date_range) in enumerate(self.cv.split(group_ordered_array)):
                if idx == cv_split_index:
                    break
        
        return train, test, date_range
    
    def record_predictions(self, group_key, cv_split_index, prediction_result, test_data, test_idx):
        """memmap['groups'][group_key]['groups'][group_key_innder]['arrays'][filepath, dtype, shape]

        """

        if self.verbose: start_time = time.time()

        test_data_prediction = test_data.merge(prediction_result, on=constants.EF_UUID_NAME, how='inner')

        predictions_array = test_data_prediction[[constants.EF_UUID_NAME, constants.EF_PREDICTION_NAME]]
        predictions_array = predictions_array.values.astype(np.float64)

        filename = '__'.join((group_key, str(cv_split_index))) + '.npy'
        filepath = os.path.join(os.getcwd(), self.task_manager.prediction_records_dirname, filename)

        # try:
        np.save(filepath, predictions_array)
        np.load(filepath)

        if self.verbose: print('Completed record_predictions:', time.time() - start_time)
        # except:
        #     pass
            # need to pass some value to indicate failure instead of unavailability!






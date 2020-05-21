import tables
import copy
import numpy as np
import pandas as pd

class TaskGraph():
    
    def __init__(self, evaluation_manager, cv):  # thinned out
        pass
        # no data in the em!
        
#       self.evaluation_manager = evaluation_manager
        # get cv object here
        self.evaluation_manager = evaluation_manager
        self.cv = cv

    def run(self, group_key, cv_split_index):    
        
        train_data, test_data = self.get_data(group_key, cv_split_index)
        self.task_graph(train_data, test_data, group_key)
        
    
    def download_hdf5_file_from_S3(self):
        pass
    
    def get_data(self, group_key, cv_split_index):
        
        self._open_hdf5_file()
        
        train_idx, test_idx = self._get_cross_validation_fold_idx(group_key, cv_split_index)
        train_data = self._read_hdf5_file(group_key, train_idx)
        test_data = self._read_hdf5_file(group_key, test_idx)
        # self._close_hdf5_file()
        
        return train_data, test_data
    
    def _open_hdf5_file(self):
        
        self.hdf5_fileobj = tables.open_file(self.evaluation_manager.hdf5_filepath, 'r')
        
    def _read_hdf5_file(self, group_key, data_idx):


        # print(self.hdf5_fileobj.open_count)
        
        missing_keys = self.hdf5_fileobj.get_node_attr('/{}'.format(group_key), 'group_attribute').missing_keys
        data_arrays = [self.hdf5_fileobj.get_node('/{}/numeric_types'.format(group_key))[data_idx, :]]
        data_colnames = copy.copy(self.hdf5_fileobj.get_node_attr('/{}'.format(group_key), 'group_attribute').numeric_keys)

        for colname in missing_keys['datetime_types']:
            tmp_array = self.hdf5_fileobj.get_node('/{}/{}'.format(group_key, colname))[data_idx]
            data_arrays.append(tmp_array.reshape(-1, 1))
            data_colnames.append(colname)

        data_array = np.hstack(data_arrays)
        pdf = pd.DataFrame(data_array, columns=data_colnames)

        for i in range(len(missing_keys['datetime_types'])):
            pdf.iloc[:, i-1] = pd.to_datetime(pdf.iloc[:, i-1])

        for colname in missing_keys['str_types']:
            tmp_array = data_array = self.hdf5_fileobj.get_node('/{}/{}'.format(group_key, colname))[data_idx]
            tmp_array = tmp_array.astype(str)
            pdf[colname] = tmp_array


            
        return pdf
    
    def _get_cross_validation_fold_idx(self, group_key, cv_split_index):
        
        if self.evaluation_manager.orderby:  # have another parameter to check orderby needs to happen...
            # by cv scheme itself!
            
            # need to add random state
            group_ordered_array = self.hdf5_fileobj.get_node('/{}/orderby_array'.format(group_key))[:]
            # cv = DateRollingWindowSplit(self.evaluation_manager.train_window, 
            #                             self.evaluation_manager.test_window, 
            #                             group_ordered_array)
            
            for idx, (train, test) in enumerate(self.cv.split(group_ordered_array)):
                if idx == cv_split_index:
                    break
        
        return train, test
    
    def _close_hdf5_file(self):
        self.hdf5_fileobj.close()
        
    
    def task_graph(self, train_data, test_data, group_key):  # groupkey is redundant info get rid of it
        
        configs = self.evaluation_manager.user_configs
        
        preprocessed_train_data = self.evaluation_manager.preprocess_train_data(
            train_data, 
            configs)
        
        # trained_estimator = self.evaluation_manager.model_fit(
        #     preprocessed_train_data, 
        #     self.evaluation_manager.hyperparameters, 
        #     self.evaluation_manager.estimator,
        #     self.evaluation_manager.feature_names[group_key],
        #     self.evaluation_manager.target_name)
        
        # preprocessed_test_data = self.evaluation_manager.preprocess_test_data(
        #     test_data, 
        #     preprocessed_train_data, 
        #     configs)
        
        # prediction_result = self.evaluation_manager.model_predict(
        #     preprocessed_test_data, 
        #     trained_estimator, 
        #     self.evaluation_manager.feature_names[group_key],
        #     self.evaluation_manager.target_name)
        
        # evaluation_result = self.evaluation_manager.evaluate_prediction(
        #     preprocessed_test_data, 
        #     prediction_result)
        
from ._evaluation_engine.dask_futures import MultiThreadTaskQueue
from ._evaluation_engine.dask_futures import DualClientFuture
from ._evaluation_engine.dask_futures import ClientFuture

from ._evaluation_engine.data_loader import load_local_data
from ._evaluation_engine.data_loader import upload_local_data
from ._evaluation_engine.data_loader import download_local_data
from ._evaluation_engine.data_loader import upload_remote_data
from ._evaluation_engine.data_loader import download_remote_data

from evaluation_framework.utils.objectIO_utils import save_obj
from evaluation_framework.utils.objectIO_utils import load_obj

from evaluation_framework.utils.memmap_utils import write_memmap
from evaluation_framework.utils.memmap_utils import read_memmap

from ._evaluation_engine.cross_validation_split import get_cv_splitter

from .task_graph import TaskGraph

import os
import pandas as pd
from collections import namedtuple


TASK_REQUIRED_KEYWORDS = [
    'memmap_root_dirname',
    'user_configs',
    'preprocess_train_data',
    'model_fit',
    'preprocess_test_data',
    'model_predict',
    'hyperparameters',
    'estimator',
    'feature_names',
    'target_name',
    'evaluate_prediction',
    'orderby',
    'return_predictions',
    'S3_path',
    'prediction_records_dirname']

TaskManager = namedtuple('TaskManager', TASK_REQUIRED_KEYWORDS)


class EvaluationEngine():

    def run_evaluation(self, evaluation_manager):
# 
        os.makedirs(evaluation_manager.local_directory_path)
        os.chdir(evaluation_manager.local_directory_path)
        
        # dl = DataLoader(evaluation_manager=evaluation_manager)
        load_local_data(evaluation_manager)

        task_manager = TaskManager(
            **{k: v for k, v in evaluation_manager.__dict__.items() 
            if k in TASK_REQUIRED_KEYWORDS})
        
        # need condition to open yarn or local!
        if evaluation_manager.S3_path:

            pass
        else:
            self.dask_client = ClientFuture(local_client_n_workers=4, 
                                   local_client_threads_per_worker=2)

            self.dask_client.get_dashboard_link()
            
            # need thread calculation: sum of available workers
            self.taskq = MultiThreadTaskQueue(num_threads=4)
            
        memmap_map_filepath = os.path.join(evaluation_manager.memmap_root_dirpath, 'memmap_map')
        memmap_map = load_obj(memmap_map_filepath)
        
        for group_key in memmap_map['attributes']['sorted_group_keys']:

            if evaluation_manager.orderby:

                filepath = memmap_map['groups'][group_key]['arrays']['orderby_array']['filepath']
                dtype = memmap_map['groups'][group_key]['arrays']['orderby_array']['dtype']
                shape = memmap_map['groups'][group_key]['arrays']['orderby_array']['shape']
                group_ordered_array = read_memmap(filepath, dtype, shape)

                cv = get_cv_splitter(
                	evaluation_manager.cross_validation_scheme, 
                	evaluation_manager.train_window, 
                	evaluation_manager.test_window, 
                	group_ordered_array)
                n_splits = cv.get_n_splits()

                task_graph = TaskGraph(evaluation_manager, cv)

                for i in range(n_splits):
                    self.taskq.put_task(self.dask_client.submit, task_graph.run, group_key, i)

            else:
                pass  # normal cross validations

    def get_evaluation_results(self):

        self.taskq.join()

        res = self.taskq.get_results()
        res_pdf = pd.DataFrame(res, columns=['group_key', 'test_idx', 'eval_result', 'data_count'])
        return res_pdf.sort_values(by=['group_key', 'test_idx']).reset_index(drop=True)


    # def get_prediction_results(self, group_key=None):






        
        
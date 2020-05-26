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
import numpy as np
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
    'memmap_root_S3_object_name',
    'prediction_records_dirname',
    'memmap_root_dirpath',
    'cross_validation_scheme',
    'train_window',
    'test_window']

TaskManager = namedtuple('TaskManager', TASK_REQUIRED_KEYWORDS)


class EvaluationEngine():

    def __init__(self, local_client_n_workers=None, local_client_threads_per_worker=None, 
                 yarn_container_n_workers=None, yarn_container_worker_vcores=None, yarn_container_worker_memory=None,
                 n_worker_nodes=None, use_yarn_cluster=None, use_auto_config=None, instance_type=None):


        # if not (use_yarn_cluster or use_auto_config):

        #     print('If ')






        # self.use_yarn_cluster = False

        # if (self.task_manager.S3_path and 
        #     yarn_container_n_workers and 
        #     yarn_container_worker_vcores and 
        #     yarn_container_worker_memory and
        #     n_worker_nodes):

        #     self.use_yarn_cluster = True

        # if (self.local_client_n_workers and self.local_client_threads_per_worker):


        self.local_client_n_workers = local_client_n_workers
        self.local_client_threads_per_worker = local_client_threads_per_worker 
        self.yarn_container_n_workers = yarn_container_n_workers
        self.yarn_container_worker_vcores = yarn_container_worker_vcores
        self.yarn_container_worker_memory = yarn_container_worker_memory
        self.n_worker_nodes = n_worker_nodes






    def run_evaluation(self, evaluation_manager):

        self.data = evaluation_manager.data

        os.makedirs(evaluation_manager.local_directory_path)
        os.chdir(evaluation_manager.local_directory_path)
        
        print("Preparing local data")
        memmap_map = load_local_data(evaluation_manager)

        # evaluation_manager is too bulky to travel across network
        self.task_manager = TaskManager(
            **{k: v for k, v in evaluation_manager.__dict__.items() 
            if k in TASK_REQUIRED_KEYWORDS})

        # need condition to open yarn or local!
        if self.task_manager.S3_path:

            upload_local_data(self.task_manager)

            self.dask_client = DualClientFuture(local_client_n_workers=self.local_client_n_workers, 
                               local_client_threads_per_worker=self.local_client_threads_per_worker, 
                               yarn_client_n_workers=self.yarn_container_n_workers*self.n_worker_nodes, 
                               yarn_client_worker_vcores=self.yarn_container_worker_vcores, 
                               yarn_client_worker_memory=self.yarn_container_worker_memory)

            self.dask_client.submit_per_node(download_local_data, self.task_manager)

            num_threads = self.local_client_n_workers + self.yarn_container_n_workers*self.n_worker_nodes

        else:
            self.dask_client = ClientFuture(local_client_n_workers=self.local_client_n_workers, 
                                   local_client_threads_per_worker=self.local_client_threads_per_worker)

            self.dask_client.get_dashboard_link()
            
            num_threads = self.local_client_n_workers
        
        self.taskq = MultiThreadTaskQueue(num_threads=num_threads)
                    
        memmap_map_filepath = os.path.join(self.task_manager.memmap_root_dirpath, 'memmap_map')
        memmap_map = load_obj(memmap_map_filepath)
        
        for group_key in memmap_map['attributes']['sorted_group_keys']:

            if self.task_manager.orderby:

                filepath = os.path.join(memmap_map['root_dirpath'], memmap_map['groups'][group_key]['arrays']['orderby_array']['filepath'])
                dtype = memmap_map['groups'][group_key]['arrays']['orderby_array']['dtype']
                shape = memmap_map['groups'][group_key]['arrays']['orderby_array']['shape']
                group_ordered_array = read_memmap(filepath, dtype, shape)

                cv = get_cv_splitter(
                	self.task_manager.cross_validation_scheme, 
                	self.task_manager.train_window, 
                	self.task_manager.test_window, 
                	group_ordered_array)
                n_splits = cv.get_n_splits()

                task_graph = TaskGraph(self.task_manager, cv)

                for i in range(n_splits):
                    self.taskq.put_task(self.dask_client.submit, task_graph.run, group_key, i)

            else:
                pass  # normal cross validations

    def get_evaluation_results(self):

        self.taskq.join()

        res = self.taskq.get_results()
        res_pdf = pd.DataFrame(res, columns=['group_key', 'test_idx', 'eval_result', 'data_count'])
        return res_pdf.sort_values(by=['group_key', 'test_idx']).reset_index(drop=True)

    def get_prediction_results(self, group_key=None):

        self.taskq.join()

        prediction_dirpath = os.path.join(os.getcwd(), self.task_manager.prediction_records_dirname)
        prediction_filenames = os.listdir(prediction_dirpath)
        prediction_filepaths = [os.path.join(prediction_dirpath, elem) for elem in prediction_filenames]

        prediction_array = np.vstack([np.load(elem) for elem in prediction_filepaths])
        prediction_array = prediction_array[prediction_array[:, 0].argsort()]

        prediction_pdf = pd.DataFrame(prediction_array, columns=['specialEF_float32_UUID', 'specialEF_float32_predictions'])
        prediction_pdf.set_index('specialEF_float32_UUID', inplace=True)
        prediction_pdf = prediction_pdf.reindex(range(0, len(self.data)), fill_value=np.nan)
        self.data['specialEF_float32_predictions'] = prediction_pdf['specialEF_float32_predictions']
        return self.data   


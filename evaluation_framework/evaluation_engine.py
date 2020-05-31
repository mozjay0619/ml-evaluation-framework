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


INSTANCE_TYPES = {
    'm4.large': {'vCPU': 2, 'Mem': 8},
    'm4.xlarge': {'vCPU': 4, 'Mem': 16}, 
    'm4.2xlarge': {'vCPU': 8, 'Mem': 32},
    'm4.4xlarge': {'vCPU': 16, 'Mem': 64},
    'm4.10xlarge': {'vCPU': 40, 'Mem': 160},
    'm4.16xlarge': {'vCPU': 64, 'Mem': 256}, 
    
    'c4.large': {'vCPU': 2, 'Mem': 3.75},
    'c4.xlarge': {'vCPU': 4, 'Mem': 7.5},
    'c4.2xlarge': {'vCPU': 8, 'Mem': 15},
    'c4.4xlarge': {'vCPU': 16, 'Mem': 30},
    'c4.8xlarge': {'vCPU': 36, 'Mem': 60}, 
    
    'r4.large': {'vCPU': 2, 'Mem': 15.25},
    'r4.xlarge': {'vCPU': 4, 'Mem': 30.5},
    'r4.2xlarge': {'vCPU': 8, 'Mem': 61}, 
    'r4.4xlarge': {'vCPU': 16, 'Mem': 122},
    'r4.8xlarge': {'vCPU': 32, 'Mem': 244},
    'r4.16xlarge': {'vCPU': 64, 'Mem': 488}}

DASK_RESOURCE_PARAMETERS = [
    'local_client_n_workers', 
    'local_client_threads_per_worker', 
    'yarn_container_n_workers',
    'yarn_container_worker_vcores', 
    'yarn_container_worker_memory', 
    'n_worker_nodes']

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
                 n_worker_nodes=None, use_yarn_cluster=None, use_auto_config=None, instance_type=None, parallelize=True,
                 verbose=True):

        self.parallelize = parallelize
        
        if (local_client_n_workers is not None and 
            local_client_threads_per_worker is not None):
            
            local_client_resources_set = True
            yarn_client_resources_set = False
            
            if (yarn_container_n_workers is not None and 
                yarn_container_worker_vcores is not None and
                yarn_container_worker_memory is not None and
                n_worker_nodes is not None):
                
                yarn_client_resources_set = True
        
        if use_auto_config is None:
            print('\u2714 Set [ use_auto_config ] to True in order to automatically configure Dask resources.\n')
        
        if use_yarn_cluster is None:
            print('\u2714 Set [ use_yarn_cluster ] to True in order to leverage Yarn cluster.\n')
            
        if use_auto_config is None:
            print('You can also manually configure resources by providing arguments for the following parameters:\n\n'
                      '\u25BA {}'.format('  '.join(DASK_RESOURCE_PARAMETERS[0:4])))
            print('\n  ' + '  '.join(DASK_RESOURCE_PARAMETERS[4:]))

        if (use_auto_config is None) or (use_yarn_cluster is None):
            print('\nOptional argument(s):\n\n\u25BA {}'.format('instance_type'))#.join(self.optional_args)))
        
        if use_auto_config:
            
            if use_yarn_cluster:
            
                if (instance_type is None or n_worker_nodes is None):
                    print('\u2714 In order to auto config yarn cluster, please provide the [ instance_type ] and [ n_worker_nodes ].')
                    print('\nEX: instance_type="m4.2xlarge", n_worker_nodes=3 (excluding the master node)')

                    print('\nAvailable [ instance_type ] options: ')
                    print('\n\u25BA {}'.format('  '.join(list(INSTANCE_TYPES.keys())[0:6])))
                    print('\n  ' + '  '.join(list(INSTANCE_TYPES.keys())[6:11]))
                    print('\n  ' + '  '.join(list(INSTANCE_TYPES.keys())[11:]))

                else:
                    num_physical_cores = int(INSTANCE_TYPES[instance_type]['vCPU']/2)
                    off_set = 2 if num_physical_cores>=8 else 1
                    local_client_n_workers = max(1, num_physical_cores - off_set)
                    local_client_threads_per_worker = 2
                    yarn_container_n_workers = num_physical_cores - 1
                    yarn_container_worker_vcores = 2
                    yarn_container_worker_memory = INSTANCE_TYPES[instance_type]['Mem'] - 1.5

            else:
                local_client_n_workers = psutil.cpu_count(logical=False)
                local_client_threads_per_worker = int(psutil.cpu_count(logical=True)/num_physical_cores)
                
        self.use_yarn_cluster = use_yarn_cluster

        self.local_client_n_workers = local_client_n_workers
        self.local_client_threads_per_worker = local_client_threads_per_worker 
        self.yarn_container_n_workers = yarn_container_n_workers
        self.yarn_container_worker_vcores = yarn_container_worker_vcores
        self.yarn_container_worker_memory = yarn_container_worker_memory
        self.n_worker_nodes = n_worker_nodes

        self.verbose = verbose

    def run_evaluation(self, evaluation_manager):

        self.data = evaluation_manager.data

        if self.use_yarn_cluster and evaluation_manager.S3_path is None:
            raise ValueError('if [ use_yarn_cluster ] is set to True, you must provide [ S3_path ] to EvaluationManager object.')

        os.makedirs(evaluation_manager.local_directory_path)
        os.chdir(evaluation_manager.local_directory_path)
        
        print("\u2714 Preparing local data...                ", end="", flush=True)
        memmap_map = load_local_data(evaluation_manager)
        print('Completed!')

        # evaluation_manager is too bulky to travel across network
        self.task_manager = TaskManager(
            **{k: v for k, v in evaluation_manager.__dict__.items() 
            if k in TASK_REQUIRED_KEYWORDS})

        # need condition to open yarn or local!
        if self.use_yarn_cluster:

            print("\u2714 Uploading local data to S3 bucket...   ", end="", flush=True)
            upload_local_data(self.task_manager)
            print('Completed!')

            print("\u2714 Starting Dask client...                ", end="", flush=True)
            self.dask_client = DualClientFuture(local_client_n_workers=self.local_client_n_workers, 
                               local_client_threads_per_worker=self.local_client_threads_per_worker, 
                               yarn_client_n_workers=self.yarn_container_n_workers*self.n_worker_nodes, 
                               yarn_client_worker_vcores=self.yarn_container_worker_vcores, 
                               yarn_client_worker_memory=self.yarn_container_worker_memory)
            print('Completed!')

            print("\u2714 Preparing data on remote workers...    ", end="", flush=True)
            self.dask_client.submit_per_node(download_local_data, self.task_manager)
            print('Completed!')

            num_threads = self.local_client_n_workers + self.yarn_container_n_workers*self.n_worker_nodes

        else:

            print("\u2714 Starting Dask client...                ", end="", flush=True)
            self.dask_client = ClientFuture(local_client_n_workers=self.local_client_n_workers, 
                                   local_client_threads_per_worker=self.local_client_threads_per_worker)
            print('Completed!')
            
            num_threads = self.local_client_n_workers
        
        self.taskq = MultiThreadTaskQueue(num_threads=num_threads)

        if self.verbose:
            print('thread size: {}'.format(num_threads))
                    
        memmap_map_filepath = os.path.join(self.task_manager.memmap_root_dirpath, 'memmap_map')
        memmap_map = load_obj(memmap_map_filepath)
        
        print("\n\u23F3 Starting evaluations...         ", end="", flush=True)
        self.dask_client.get_dashboard_link()
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

                    if self.parallelize:
                        self.taskq.put_task(self.dask_client.submit, task_graph.run, group_key, i)
                    else:
                        task_graph.run(group_key, i)

            else:
                pass  # normal cross validations

    def get_evaluation_results(self):

        self.taskq.join()

        res = self.taskq.get_results()
        res_pdf = pd.DataFrame(res, columns=['group_key', 'test_idx', 'eval_result', 'data_count'])
        return res_pdf.sort_values(by=['group_key', 'test_idx']).reset_index(drop=True)

    def get_prediction_results(self, group_key=None):

        self.taskq.join()

        if self.use_yarn_cluster:

            print("\n\u2714 Uploading remote prediction results...   ", end="", flush=True)
            self.dask_client.submit_per_node(upload_remote_data, self.task_manager)
            print('Completed!')

            print("\n\u2714 Downloading remote prediction results... ", end="", flush=True)
            download_remote_data(self.task_manager)
            print('Completed!')

        prediction_dirpath = os.path.join(os.getcwd(), self.task_manager.prediction_records_dirname)
        prediction_filenames = os.listdir(prediction_dirpath)
        prediction_filepaths = [os.path.join(prediction_dirpath, elem) for elem in prediction_filenames]

        prediction_array = np.vstack([np.load(elem) for elem in prediction_filepaths])
        prediction_array = prediction_array[prediction_array[:, 0].argsort()]

        prediction_pdf = pd.DataFrame(prediction_array, columns=['specialEF_float32_UUID', 'specialEF_float32_predictions'])
        prediction_pdf.set_index('specialEF_float32_UUID', inplace=True)
        prediction_pdf = prediction_pdf.reindex(range(0, len(self.data)), fill_value=np.nan)
        self.data['specialEF_float32_predictions'] = prediction_pdf['specialEF_float32_predictions']
        self.data.drop(labels='specialEF_float32_UUID', axis=1, inplace=True)
        return self.data   


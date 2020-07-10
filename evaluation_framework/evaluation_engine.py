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
from evaluation_framework import constants

import HMF

import os
import pandas as pd
import numpy as np
from collections import namedtuple
import psutil
import shutil
import time


# INSTANCE_TYPES = {
#     'm4.large': {'vCPU': 2, 'Mem': 8},
#     'm4.xlarge': {'vCPU': 4, 'Mem': 16}, 
#     'm4.2xlarge': {'vCPU': 8, 'Mem': 32},
#     'm4.4xlarge': {'vCPU': 16, 'Mem': 64},
#     'm4.10xlarge': {'vCPU': 40, 'Mem': 160},
#     'm4.16xlarge': {'vCPU': 64, 'Mem': 256}, 
    
#     'c4.large': {'vCPU': 2, 'Mem': 3.75},
#     'c4.xlarge': {'vCPU': 4, 'Mem': 7.5},
#     'c4.2xlarge': {'vCPU': 8, 'Mem': 15},
#     'c4.4xlarge': {'vCPU': 16, 'Mem': 30},
#     'c4.8xlarge': {'vCPU': 36, 'Mem': 60}, 
    
#     'r4.large': {'vCPU': 2, 'Mem': 15.25},
#     'r4.xlarge': {'vCPU': 4, 'Mem': 30.5},
#     'r4.2xlarge': {'vCPU': 8, 'Mem': 61}, 
#     'r4.4xlarge': {'vCPU': 16, 'Mem': 122},
#     'r4.8xlarge': {'vCPU': 32, 'Mem': 244},
#     'r4.16xlarge': {'vCPU': 64, 'Mem': 488}}

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
    
    'r4.large': {'vCPU': 2, 'Mem': 15.25 - 3},
    'r4.xlarge': {'vCPU': 4, 'Mem': 30.5 - 6},
    'r4.2xlarge': {'vCPU': 8, 'Mem': 61 - 12}, 
    'r4.4xlarge': {'vCPU': 16, 'Mem': 122 - 25},
    'r4.8xlarge': {'vCPU': 32, 'Mem': 244 - 50},
    'r4.16xlarge': {'vCPU': 64, 'Mem': 488 - 50}}

DEFAULT_LARGE_INSTANCE_WORKER_VCORES = 4
DEFAULT_SMALL_INSTANCE_WORKER_VCORES = 2

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
    'min_train_window',
    'test_window',
    'evaluation_task_dirname', 
    'evaluation_task_dirpath',
    'job_uuid']

DEBUG_MODE_MAXITER = 10

TaskManager = namedtuple('TaskManager', TASK_REQUIRED_KEYWORDS)


class EvaluationEngine():

    def __init__(self, local_client_n_workers=None, local_client_threads_per_worker=None, 
                 yarn_container_n_workers=None, yarn_container_worker_vcores=None, yarn_container_worker_memory=None,
                 n_worker_nodes=None, use_yarn_cluster=None, use_auto_config=None, instance_type=None,
                 verbose=False):
        
        self.verbose = verbose
        
        self.validate_dask_resource_configs(
            local_client_n_workers, local_client_threads_per_worker, 
            yarn_container_n_workers, yarn_container_worker_vcores, yarn_container_worker_memory,
            n_worker_nodes, use_yarn_cluster, use_auto_config, instance_type)

        self.has_dask_client = False
        
    def run_evaluation(self, evaluation_manager, debug_mode=False):

        self.data = evaluation_manager.data

        if self.use_yarn_cluster and evaluation_manager.S3_path is None:
            raise ValueError('if [ use_yarn_cluster ] is set to True, you must provide [ S3_path ] to EvaluationManager object.')

        if os.path.exists(evaluation_manager.evaluation_task_dirpath):
            print('\u2757 Removing duplicate evaluation_task_dirpath\n')
            shutil.rmtree(evaluation_manager.evaluation_task_dirpath)
        os.makedirs(evaluation_manager.evaluation_task_dirpath)
        os.chdir(evaluation_manager.evaluation_task_dirpath)
        # by not removing the local_directory_path (root) but just the task specific dir, 
        # we can ensure re-runnability of the current evaluation task.
        # if EM were to be redefined, another task dir will be created.
        # the change of directory is required for sharing methods across yarn and local clients
        # also, need to start dask AFTER the change in directory 


        if(not debug_mode):
            
            if not self.has_dask_client:
                self.start_dask_client()
                self.has_dask_client = True
            else:
                self.stop_dask_client()
                self.start_dask_client()
                self.has_dask_client = True
                
        print("\u2714 Preparing local data...            ", end="", flush=True)
        print()
        # self.memmap_map = load_local_data(evaluation_manager)
        load_local_data(evaluation_manager)
        # print('Completed!')

        root_dirpath = os.path.join(os.getcwd(), evaluation_manager.memmap_root_dirname)
        self.f = HMF.open_file(root_dirpath, mode='r+')
        
        # evaluation_manager is too bulky to travel across network
        self.task_manager = TaskManager(
            **{k: v for k, v in evaluation_manager.__dict__.items() 
            if k in TASK_REQUIRED_KEYWORDS})
        
        if self.use_yarn_cluster:
            
            print("\u2714 Uploading local data to S3 bucket...   ", end="", flush=True)
            upload_local_data(self.task_manager)
            print('Completed!')
            
            print("\u2714 Preparing data on remote workers...   ", end="", flush=True)
            self.dask_client.submit_per_node(download_local_data, self.task_manager)
            print('Completed!')
        
        if debug_mode:
            print('\nRunning on debug mode!')

            ###################################################################################################################
            ###################################################### DEBUG MODE #################################################
            ###################################################################################################################

            iter_count = 0

            start_time = time.time()

            total_splits = 0
            for group_key in self.f.get_node_attr('/', key='sorted_group_keys'):

                if self.task_manager.orderby:

                    group_orderby_array = self.get_group_orderby_array(group_key)

                    cv = get_cv_splitter(
                        self.task_manager.cross_validation_scheme, 
                        self.task_manager.train_window, 
                        self.task_manager.test_window,
                        self.task_manager.min_train_window,
                        group_orderby_array)
                    total_splits += cv.get_n_splits()


            print(total_splits)
            print(time.time() - start_time)


            for group_key in self.f.get_node_attr('/', key='sorted_group_keys'):

                if self.task_manager.orderby:

                    group_orderby_array = self.get_group_orderby_array(group_key)

                    cv = get_cv_splitter(
                        self.task_manager.cross_validation_scheme, 
                        self.task_manager.train_window, 
                        self.task_manager.test_window,
                        self.task_manager.min_train_window,
                        group_orderby_array)
                    n_splits = cv.get_n_splits()

                    task_graph = TaskGraph(self.task_manager, cv)

                    for i in range(n_splits):

                        task_graph.run(group_key, i)

                        iter_count += 1

                        if(iter_count == DEBUG_MODE_MAXITER):
                            print('\nReached DEBUG_MODE_MAXITER stopping')
                            return


                else:
                    pass

            ###################################################################################################################
            ###################################################### DEBUG MODE #################################################
            ###################################################################################################################


            return 

        print("\n\u23F3 Starting evaluations...   ")
        self.dask_client.get_dashboard_link()
        # for group_key in self.memmap_map['attributes']['sorted_group_keys']:
        for group_key in self.f.get_node_attr('/', key='sorted_group_keys'):

            if self.task_manager.orderby:

                group_orderby_array = self.get_group_orderby_array(group_key)

                cv = get_cv_splitter(
                    self.task_manager.cross_validation_scheme, 
                    self.task_manager.train_window, 
                    self.task_manager.test_window,
                    self.task_manager.min_train_window,
                    group_orderby_array)
                n_splits = cv.get_n_splits()

                task_graph = TaskGraph(self.task_manager, cv)

                for i in range(n_splits):

                    self.taskq.put_task(self.dask_client.submit, task_graph.run, group_key, i)
                    
            else:
                pass  # normal cross validations

        os.chdir(evaluation_manager.initial_dirpath)
        
    def get_group_orderby_array(self, group_key):

        


        
        # filepath = os.path.join(self.memmap_map['root_dirpath'], self.memmap_map['groups'][group_key]['arrays']['orderby_array']['filepath'])
        # dtype = self.memmap_map['groups'][group_key]['arrays']['orderby_array']['dtype']
        # shape = self.memmap_map['groups'][group_key]['arrays']['orderby_array']['shape']
        # group_orderby_array = read_memmap(filepath, dtype, shape)

        group_orderby_array = self.f.get_array('/{}/orderby_array'.format(group_key))
        return group_orderby_array

    def start_dask_client(self):
        
        if self.use_yarn_cluster:

            print("\u2714 Starting Dask client...            ", end="", flush=True)
            self.dask_client = DualClientFuture(local_client_n_workers=self.local_client_n_workers, 
                               local_client_threads_per_worker=self.local_client_threads_per_worker, 
                               yarn_client_n_workers=self.yarn_container_n_workers*self.n_worker_nodes, 
                               yarn_client_worker_vcores=self.yarn_container_worker_vcores, 
                               yarn_client_worker_memory=self.yarn_container_worker_memory)
            print('Completed!')

            self.dask_local_client = self.dask_client.local_client
            self.dask_yarn_client = self.dask_client.yarn_client

            num_threads = self.local_client_n_workers + self.yarn_container_n_workers*self.n_worker_nodes

        else:

            print("\u2714 Starting Dask client...            ", end="", flush=True)
            self.dask_client = ClientFuture(local_client_n_workers=self.local_client_n_workers, 
                                   local_client_threads_per_worker=self.local_client_threads_per_worker)
            print('Completed!')
            
            self.dask_local_client = self.dask_client.local_client
            self.dask_yarn_client = None
            
            num_threads = self.local_client_n_workers
        
        self.taskq = MultiThreadTaskQueue(num_threads=num_threads)
        
        if self.verbose:
            print('thread size: {}'.format(num_threads))
        
    def stop_dask_client(self):
        
        if self.use_yarn_cluster:
            self.dask_client.local_client.close()
            self.dask_client.local_cluster.close()
            
            self.dask_client.yarn_client.close()
            self.dask_client.yarn_cluster.close()
            
        else:
            self.dask_client.local_client.close()
            self.dask_client.local_cluster.close()
        
    def validate_dask_resource_configs(self, local_client_n_workers, local_client_threads_per_worker, 
        yarn_container_n_workers, yarn_container_worker_vcores, yarn_container_worker_memory,
        n_worker_nodes, use_yarn_cluster, use_auto_config, instance_type):

        local_client_resources_set = False
        yarn_client_resources_set = False

        if (local_client_n_workers is not None and 
            local_client_threads_per_worker is not None):
            local_client_resources_set = True

            if (yarn_container_n_workers is not None and 
                yarn_container_worker_vcores is not None and
                yarn_container_worker_memory is not None and
                n_worker_nodes is not None):
                yarn_client_resources_set = True

        if local_client_resources_set:
            self.local_client_n_workers = local_client_n_workers
            self.local_client_threads_per_worker = local_client_threads_per_worker

            if yarn_client_resources_set:
                self.yarn_container_n_workers = yarn_container_n_workers
                self.yarn_container_worker_vcores = yarn_container_worker_vcores
                self.yarn_container_worker_memory = yarn_container_worker_memory
                self.n_worker_nodes = n_worker_nodes
            return

        if use_auto_config is None:
            print('\u2714 Set [ use_auto_config ] to True in order to automatically configure Dask resources.\n')

        if use_yarn_cluster is None:
            print('\u2714 Set [ use_yarn_cluster ] to True in order to leverage Yarn cluster.\n')

        if use_auto_config is None:
            print('\u27AA You can also manually configure resources by providing arguments for the following '
                      'parameters:\n\n\u25BA {}'.format('  '.join(DASK_RESOURCE_PARAMETERS[0:4])))
            print('\n  ' + '  '.join(DASK_RESOURCE_PARAMETERS[4:]))

        if (use_auto_config is None) or (use_yarn_cluster is None):
            print('\nOptional argument(s):\n\n\u25BA {}'.format('instance_type'))

        if use_auto_config:
            if use_yarn_cluster:
                self.use_yarn_cluster = True
                
                if (instance_type is None or n_worker_nodes is None):
                    print('\u2714 In order to auto config yarn cluster, please provide the [ instance_type ] '
                          'and [ n_worker_nodes ].')
                    print('\nEX: instance_type="m4.2xlarge", n_worker_nodes=3 (excluding the master node)')
                    print('\nAvailable [ instance_type ] options: ')
                    print('\n\u25BA {}'.format('  '.join(list(INSTANCE_TYPES.keys())[0:6])))
                    print('\n  ' + '  '.join(list(INSTANCE_TYPES.keys())[6:11]))
                    print('\n  ' + '  '.join(list(INSTANCE_TYPES.keys())[11:]))
                    return

                else:
                    num_physical_cores = int(INSTANCE_TYPES[instance_type]['vCPU']/2)
                    num_virtual_cores = int(INSTANCE_TYPES[instance_type]['vCPU'])
                    available_memory = int(INSTANCE_TYPES[instance_type]['Mem'] - 4)
                    # 2 GB claimed by client + 2 GB claimed by scheduler in a node

                    large_instance = num_physical_cores>=8

                    if large_instance:

                        local_offset = 4
                        self.local_client_threads_per_worker = DEFAULT_LARGE_INSTANCE_WORKER_VCORES
                        self.local_client_n_workers = int((num_virtual_cores - 
                                                      local_offset)/self.local_client_threads_per_worker)
                
                        yarn_offset = 2
                        self.yarn_container_worker_vcores = DEFAULT_LARGE_INSTANCE_WORKER_VCORES
                        self.yarn_container_n_workers = int((num_virtual_cores - 
                                                             yarn_offset)/self.yarn_container_worker_vcores)
                        self.yarn_container_worker_memory = str(int((available_memory - 
                                                                1.5)/self.yarn_container_n_workers)) + ' GB'
                        self.yarn_container_worker_memory = str(int((available_memory - 
                                                                1.5)/self.yarn_container_n_workers)) + ' GB'

                    else:
                        local_offset = 2
                        self.local_client_threads_per_worker = DEFAULT_SMALL_INSTANCE_WORKER_VCORES
                        self.local_client_n_workers = int(max(1, num_virtual_cores - 
                                                         local_offset)/self.local_client_threads_per_worker)

                        yarn_offset = 2
                        self.yarn_container_worker_vcores = DEFAULT_SMALL_INSTANCE_WORKER_VCORES
                        self.yarn_container_n_workers = int(max(1, num_virtual_cores - 
                                                           yarn_offset)/self.yarn_container_worker_vcores)
                        self.yarn_container_worker_memory = str(int((available_memory - 
                                                                1.5)/self.yarn_container_n_workers)) + ' GB'

                self.n_worker_nodes = n_worker_nodes
                
                print('[ aws instance configurations ]')
                print('instance vcores: {}'.format(INSTANCE_TYPES[instance_type]['vCPU']))
                print('instance memory: {} GB'.format(INSTANCE_TYPES[instance_type]['Mem']))
                print('n_worker_nodes: {}'.format(self.n_worker_nodes))
                print('[ dask configurations ]')
                print('local_client_n_workers: {}'.format(self.local_client_n_workers))
                print('local_client_threads_per_worker: {}'.format(self.local_client_threads_per_worker))
                print('yarn_container_n_workers: {}'.format(self.yarn_container_n_workers))
                print('yarn_container_worker_vcores: {}'.format(self.yarn_container_worker_vcores))
                print('yarn_container_worker_memory: {}'.format(self.yarn_container_worker_memory))

            else:
                self.use_yarn_cluster = False
                
                self.local_client_n_workers = psutil.cpu_count(logical=False)
                self.local_client_threads_per_worker = int(psutil.cpu_count(logical=True)/self.local_client_n_workers)
                
                print('[ dask configurations ]')
                print('local_client_n_workers: {}'.format(self.local_client_n_workers))
                print('local_client_threads_per_worker: {}'.format(self.local_client_threads_per_worker))
                
    def get_evaluation_results(self):

        self.taskq.join()

        res = self.taskq.get_results()
        res_pdf = pd.DataFrame(res, columns=['group_key', 'test_idx', 'eval_result', 'data_count'])
        return res_pdf.sort_values(by=['group_key', 'test_idx']).reset_index(drop=True)

        return res

    def get_prediction_results(self, group_key=None):


        # print(os.getcwd())

        self.taskq.join()

        if self.use_yarn_cluster:

            print("\n\u2714 Uploading remote prediction results...   ", end="", flush=True)
            self.dask_client.submit_per_node(upload_remote_data, self.task_manager)
            print('Completed!')

            print("\n\u2714 Downloading remote prediction results... ", end="", flush=True)
            download_remote_data(self.task_manager)
            print('Completed!')

        prediction_dirpath = os.path.join(self.task_manager.evaluation_task_dirpath, self.task_manager.prediction_records_dirname)
        prediction_filenames = os.listdir(prediction_dirpath)

        prediction_filepaths = [os.path.join(prediction_dirpath, elem) for elem in prediction_filenames]

        prediction_array = np.vstack([np.load(elem) for elem in prediction_filepaths])
        prediction_array = prediction_array[prediction_array[:, 0].argsort()]

        prediction_pdf = pd.DataFrame(prediction_array, columns=[constants.EF_UUID_NAME, constants.EF_PREDICTION_NAME])
        prediction_pdf.set_index(constants.EF_UUID_NAME, inplace=True)
        prediction_pdf = prediction_pdf.reindex(range(0, len(self.data)), fill_value=np.nan)
        self.data[constants.EF_PREDICTION_NAME] = prediction_pdf[constants.EF_PREDICTION_NAME]
        
        return self.data.drop(labels=constants.EF_UUID_NAME, axis=1, inplace=False)


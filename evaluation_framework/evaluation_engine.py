# from .evaluation_engine_core.parallel.dask_client_future import MultiThreadTaskQueue
# from .evaluation_engine_core.parallel.dask_client_future import DualClientFuture
# from .evaluation_engine_core.parallel.dask_client_future import ClientFuture

from .evaluation_engine_core.parallel.dask_resource_configurer import DaskResourceConfigurer
from .evaluation_engine_core.parallel.dask_client import DaskClient

from .evaluation_engine_core.data_loader import DataLoader

# from .evaluation_engine_core.data_loader import load_local_data

from .evaluation_engine_core.data_transferer import upload_local_data
from .evaluation_engine_core.data_transferer import download_local_data
from .evaluation_engine_core.data_transferer import upload_remote_data
from .evaluation_engine_core.data_transferer import download_remote_data

from evaluation_framework.utils.objectIO_utils import save_obj
from evaluation_framework.utils.objectIO_utils import load_obj
from evaluation_framework.utils.memmap_utils import write_memmap
from evaluation_framework.utils.memmap_utils import read_memmap
from evaluation_framework.utils.fileIO_utils import clean_dir

from .task_graph.cross_validation_split import get_cv_splitter
from .task_graph.task_graph import TaskGraph
from evaluation_framework import constants

import HMF

import os
import pandas as pd
import numpy as np
from collections import namedtuple
import psutil
import shutil
import time
import copy




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
                 n_worker_nodes=None, use_yarn_cluster=None, use_ec2_instance=None, use_auto_config=None, instance_type=None,
                 verbose=False, use_dashboard=True):
        
        self.verbose = verbose

        self.resource_config = DaskResourceConfigurer()
        self.resource_config.validate_dask_resource_configs(
        	local_client_n_workers, local_client_threads_per_worker, 
        	yarn_container_n_workers, yarn_container_worker_vcores, yarn_container_worker_memory,
        	n_worker_nodes, use_yarn_cluster, use_ec2_instance, use_auto_config, instance_type, use_dashboard)

        self.has_dask_client = False
        self.has_prediction = False
        self.has_data_loader_scatter = False

        self.use_dashboard = use_dashboard
        
    def run_evaluation(self, evaluation_manager, debug_mode=False):

        self.has_prediction = False

        self.data = evaluation_manager.data

        # if self.use_yarn_cluster and evaluation_manager.S3_path is None:
        #     raise ValueError('if [ use_yarn_cluster ] is set to True, you must provide [ S3_path ] to EvaluationManager object.')

        if not evaluation_manager.local_data_saved:

            if os.path.exists(evaluation_manager.evaluation_task_dirpath):
                print('\u2757 Removing duplicate evaluation_task_dirpath\n')
                shutil.rmtree(evaluation_manager.evaluation_task_dirpath)
            os.makedirs(evaluation_manager.evaluation_task_dirpath)

        else:

            print('\u2757 Reusing previous evaluation data and dask cluster')

            # self.taskq.flush_results()
        
        os.chdir(evaluation_manager.evaluation_task_dirpath)
            # by not removing the local_directory_path (root) but just the task specific dir, 
            # we can ensure re-runnability of the current evaluation task.
            # if EM were to be redefined, another task dir will be created.
            # the change of directory is required for sharing methods across yarn and local clients
            # also, need to start dask AFTER the change in directory 

        if evaluation_manager.return_predictions:

            prediction_records_dirpath = os.path.join(os.getcwd(), evaluation_manager.prediction_records_dirname)
            try:
                os.makedirs(prediction_records_dirpath)
            except:
                shutil.rmtree(prediction_records_dirpath)
                os.makedirs(prediction_records_dirpath)


        if(not debug_mode):
            
            if not self.has_dask_client:
                # self.start_dask_client()

                self.dask_client = DaskClient()
                self.dask_client.start_dask_client(local_client_n_workers=self.resource_config.local_client_n_workers,
                              local_client_threads_per_worker=self.resource_config.local_client_threads_per_worker,
                              yarn_client_n_workers=self.resource_config.yarn_client_n_workers,
                              yarn_client_worker_vcores=self.resource_config.yarn_client_worker_vcores,
                              yarn_client_worker_memory=self.resource_config.yarn_client_worker_memory,
                              use_dashboard=self.use_dashboard)

                self.has_dask_client = True
            else:
                # reuse the client
                pass
                # self.stop_dask_client()
                # self.start_dask_client()
                # self.has_dask_client = True

        if not evaluation_manager.local_data_saved:
                    
            print("\u2714 Preparing local data...            ", end="", flush=True)
            print()
            # self.memmap_map = load_local_data(evaluation_manager)


            self.data_loader = DataLoader(os.path.join(os.getcwd(), evaluation_manager.memmap_root_dirname), True)
            self.data_loader.save_data(
			    evaluation_manager.data,
			    evaluation_manager.orderby,
			    evaluation_manager.groupby,
			    evaluation_manager.numeric_types,
			    evaluation_manager.missing_keys)



        else:

            pass



            # data_loader = DataLoader(os.path.join(os.getcwd(), evaluation_manager.memmap_root_dirname), False)

            # return data_loader


            # load_local_data(evaluation_manager)
            # print('Completed!')

        # root_dirpath = os.path.join(os.getcwd(), evaluation_manager.memmap_root_dirname)
        # self.f = HMF.open_file(root_dirpath, mode='r+')
        
        # evaluation_manager is too bulky to travel across network
        self.task_manager = TaskManager(
            **{k: v for k, v in evaluation_manager.__dict__.items() 
            if k in TASK_REQUIRED_KEYWORDS})

        if not evaluation_manager.local_data_saved:
            
            if self.resource_config.use_yarn_cluster:
                
                print("\u2714 Uploading local data to S3 bucket...   ", end="", flush=True)
                upload_local_data(self.task_manager)
                print('Completed!')
                
                print("\u2714 Preparing data on remote workers...   ", end="", flush=True)
                self.dask_client.submit_per_node(download_local_data, self.task_manager)
                print('Completed!')

            evaluation_manager.local_data_saved = True
            
        if debug_mode:
            print('\nRunning on debug mode!')

            ###################################################################################################################
            ###################################################### DEBUG MODE #################################################
            ###################################################################################################################

            iter_count = 0

            start_time = time.time()

            # total_splits = 0
            # for group_key in self.f.get_node_attr('/', key='sorted_group_keys'):

            #     if self.task_manager.orderby:

            #         group_orderby_array = self.get_group_orderby_array(group_key)

            #         cv = get_cv_splitter(
            #             self.task_manager.cross_validation_scheme, 
            #             self.task_manager.train_window, 
            #             self.task_manager.test_window,
            #             self.task_manager.min_train_window,
            #             group_orderby_array)
            #         total_splits += cv.get_n_splits()


            # print(total_splits)
            print(time.time() - start_time)


            for group_key in self.data_loader.f.get_sorted_group_names():

                print(group_key)

                

                if self.task_manager.orderby:

                    group_orderby_array = self.data_loader.f.get_array('/{}/orderby_array'.format(group_key))

                    cv = get_cv_splitter(
                        self.task_manager.cross_validation_scheme, 
                        self.task_manager.train_window, 
                        self.task_manager.test_window,
                        self.task_manager.min_train_window,
                        group_orderby_array)
                    n_splits = cv.get_n_splits()

                    task_graph = TaskGraph(self.task_manager, cv, verbose=self.verbose)

                    for i in range(n_splits):

                        task_graph.run(group_key, i, self.data_loader)

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

        if not self.has_data_loader_scatter:
            self.data_loader_scattered = self.dask_client.scatter(self.data_loader)[0]
            self.has_data_loader_scatter = True
        
        for group_key in self.data_loader.f.get_sorted_group_names():

            if self.task_manager.orderby:

                group_orderby_array = self.data_loader.f.get_array('/{}/orderby_array'.format(group_key))

                cv = get_cv_splitter(
                    self.task_manager.cross_validation_scheme, 
                    self.task_manager.train_window, 
                    self.task_manager.test_window,
                    self.task_manager.min_train_window,
                    group_orderby_array)
                n_splits = cv.get_n_splits()

                task_graph = TaskGraph(self.task_manager, cv)

                for i in range(n_splits):


                    self.dask_client.submit(task_graph.run, group_key, i, self.data_loader_scattered)


                    # self.taskq.put_task(self.dask_client.submit, task_graph.run, group_key, i)
                    
            else:
                pass  # normal cross validations

        os.chdir(evaluation_manager.initial_dirpath)
        

    # def start_dask_client(self):
        
    #     if self.use_yarn_cluster:

    #         print("\u2714 Starting Dask client...            ", end="", flush=True)
    #         self.dask_client = DualClientFuture(local_client_n_workers=self.local_client_n_workers, 
    #                            local_client_threads_per_worker=self.local_client_threads_per_worker, 
    #                            yarn_client_n_workers=self.yarn_container_n_workers*self.n_worker_nodes, 
    #                            yarn_client_worker_vcores=self.yarn_container_worker_vcores, 
    #                            yarn_client_worker_memory=self.yarn_container_worker_memory)
    #         print('Completed!')

    #         self.dask_local_client = self.dask_client.local_client
    #         self.dask_yarn_client = self.dask_client.yarn_client

    #         num_threads = self.local_client_n_workers + self.yarn_container_n_workers*self.n_worker_nodes

    #     else:

    #         print("\u2714 Starting Dask client...            ", end="", flush=True)
    #         self.dask_client = ClientFuture(local_client_n_workers=self.local_client_n_workers, 
    #                                local_client_threads_per_worker=self.local_client_threads_per_worker)
    #         print('Completed!')
            
    #         self.dask_local_client = self.dask_client.local_client
    #         self.dask_yarn_client = None
            
    #         num_threads = self.local_client_n_workers
        
    #     self.taskq = MultiThreadTaskQueue(num_threads=num_threads)
        
    #     if self.verbose:
    #         print('thread size: {}'.format(num_threads))
        
    # def stop_dask_client(self):
        
    #     if self.use_yarn_cluster:
    #         self.dask_client.local_client.close()
    #         self.dask_client.local_cluster.close()
            
    #         self.dask_client.yarn_client.close()
    #         self.dask_client.yarn_cluster.close()
            
    #     else:
    #         self.dask_client.local_client.close()
    #         self.dask_client.local_cluster.close()
        
    
                
    def get_evaluation_results(self):

        # self.taskq.join()

        res = copy.deepcopy(self.dask_client.get_results())

        tmp = self.data[[self.task_manager.orderby, constants.EF_ORDERBY_NAME]]
        tmp.set_index(constants.EF_ORDERBY_NAME, inplace=True)
        tmp_dict = tmp.to_dict()[self.task_manager.orderby]
        tmp_dict = {k: str(v.date()) for k, v in tmp_dict.items()}

        for idx, elem in enumerate(res):
            res[idx][-2] = [tmp_dict[_elem] for _elem in elem[-2] if _elem in tmp_dict]

        res_pdf = pd.DataFrame(res, columns=['group_key', 'test_idx', 'eval_result', 'train_size', 'test_size', 'test_dates', 'duration'])
        return res_pdf.sort_values(by=['group_key', 'test_idx']).reset_index(drop=True)

        
        # res_pdf = pd.DataFrame(res, columns=['group_key', 'test_idx', 'eval_result', 'train_size', 'test_size', 'duration'])
        # return res_pdf.sort_values(by=['group_key', 'test_idx']).reset_index(drop=True)

        # return res

    def get_evaluation_summary(self):

        res = self.get_evaluation_results()

        re_dict = {}
        for group_key, grouped_pdf in res.groupby('group_key'):
            re = np.sum(grouped_pdf['eval_result']*grouped_pdf['train_size'])/grouped_pdf['train_size'].sum()
            re_dict[group_key] = re
            
        return pd.DataFrame([(k, v) for k, v in re_dict.items()], columns=['group_key', 'eval_result'])

    def get_prediction_results(self, group_key=None):

        if not self.has_prediction:

            self.dask_client.get_results()

            if self.resource_config.use_yarn_cluster:

                print("\n\u2714 Uploading remote prediction results...   ", end="", flush=True)
                self.dask_client.submit_per_node(upload_remote_data, self.task_manager)
                print('Completed!')

                print("\n\u2714 Downloading remote prediction results... ", end="", flush=True)
                download_remote_data(self.task_manager)
                print('Completed!')

            prediction_dirpath = os.path.join(self.task_manager.evaluation_task_dirpath, self.task_manager.prediction_records_dirname)
            prediction_filenames = os.listdir(prediction_dirpath)

            prediction_filepaths = [os.path.join(prediction_dirpath, elem) for elem in prediction_filenames]
            prediction_filepaths = [elem for elem in prediction_filepaths if elem.split('.')[-1]=='npy']

            prediction_array = np.vstack([np.load(elem) for elem in prediction_filepaths])
            prediction_array = prediction_array[prediction_array[:, 0].argsort()]

            prediction_pdf = pd.DataFrame(prediction_array, columns=[constants.EF_UUID_NAME, constants.EF_PREDICTION_NAME])
            prediction_pdf.set_index(constants.EF_UUID_NAME, inplace=True)
            prediction_pdf = prediction_pdf.reindex(range(0, len(self.data)), fill_value=np.nan)
            self.data[constants.EF_PREDICTION_NAME] = prediction_pdf[constants.EF_PREDICTION_NAME]

            self.has_prediction = True
            clean_dir(prediction_dirpath)
            
            return self.data.drop(labels=[
                constants.EF_UUID_NAME, 
                constants.EF_ORDERBY_NAME,
                constants.HMF_MEMMAP_MAP_NAME,
                constants.HMF_GROUPBY_NAME], axis=1, inplace=False, errors='ignore')

        else:

            return self.data.drop(labels=[
                constants.EF_UUID_NAME, 
                constants.EF_ORDERBY_NAME,
                constants.HMF_MEMMAP_MAP_NAME,
                constants.HMF_GROUPBY_NAME], axis=1, inplace=False, errors='ignore')












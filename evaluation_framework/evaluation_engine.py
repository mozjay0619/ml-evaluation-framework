from ._evaluation_engine.dask_futures import MultiThreadTaskQueue
from ._evaluation_engine.dask_futures import DualClientFuture
from ._evaluation_engine.dask_futures import ClientFuture

from ._evaluation_engine.data_loader import DataLoader

from ._evaluation_engine.memmap_layer import read_memmap
from ._evaluation_engine.memmap_layer import load_obj

from ._evaluation_engine.split import get_cv_splitter

from ._evaluation_engine.task_graph import TaskGraph

import os
import pandas as pd



TASK_REQUIRED_FIELDS = ['estimator', 'data', 'target_name', 'feature_names', 
                   'cross_validation_scheme', 'hyperparameters',
                   'orderby', 'train_window', 'test_window', 'groupby',
                   'local_directory_path', 'S3_path',
                   
                   'str_types', 'datetime_types', 'numeric_types',
                   
                   'num_types_needed', 'missing_keys']




class EvaluationEngine():

    def run_evaluation(self, evaluation_manager):
# 
        os.makedirs(evaluation_manager.local_directory_path)
        os.chdir(evaluation_manager.local_directory_path)
        
        dl = DataLoader(evaluation_manager=evaluation_manager)
        dl.load_local_data()
        
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

                # lean_evaluation_manager = 

                task_graph = TaskGraph(evaluation_manager, cv)

                for i in range(n_splits):
                    self.taskq.put_task(self.dask_client.submit, task_graph.run, group_key, i)

            else:
                pass  # normal cross validations

        self.taskq.join()

        res = self.taskq.get_results()
        res_pdf = pd.DataFrame(res, columns=['group_key', 'test_idx', 'eval_result'])
        return res_pdf.sort_values(by=['group_key', 'test_idx']).reset_index(drop=True)


        
        
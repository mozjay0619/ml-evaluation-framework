from evaluation_framework.evaluation_engine_core.parallel.dask_client_future import MultiThreadTaskQueue
from evaluation_framework.evaluation_engine_core.parallel.dask_client_future import ClientFuture
from evaluation_framework.evaluation_engine_core.parallel.dask_client_future import DualClientFuture

class DaskClient():
    
    def __init__(self, yarn_cluster=False, multithreaded=False, ):
        
        self.multithreaded = multithreaded
        self.yarn_cluster = yarn_cluster
        
        self.futures = []
    
    def start_dask_client(self, dask_client=None,
                          local_client_n_workers=None, local_client_threads_per_worker=None,
                          yarn_client_n_workers=None, yarn_client_worker_vcores=None, 
                          yarn_client_worker_memory=None, use_dashboard=True):
        """
        The configure inputs are None default since we could pass in dask_client directly
        """
        
        if dask_client is not None:
            self.dask_client = dask_client
            return
        
        if not self.multithreaded:
            self.dask_client = ClientFuture(
                local_client_n_workers=local_client_n_workers, 
                local_client_threads_per_worker=local_client_threads_per_worker, 
                use_dashboard=use_dashboard)
            
        else:
            self.taskq = MultiThreadTaskQueue(num_threads=num_threads)
            
    def scatter(self, *args):
        
        if self.multithreaded:
            pass
        else:
            if self.yarn_cluster:
                pass
            else:
                scattered_args = self.dask_client.scatter(*args)
                return scattered_args
            
    def submit(self, func, *args, **kwargs):
        
        if self.multithreaded:
            future = self.dask_client.submit(func, *args, **kwargs)
            return future.result() 
        
        else:
            future = self.dask_client.submit(func, *args, **kwargs)
            self.futures.append(future)
            
    def get_results(self):
        
        if self.multithreaded:
            pass
        else:
            return [list(elem.result()) for elem in self.futures]
        
    def get_dashboard_link(self):
        
        self.dask_client.get_dashboard_link()
            
        




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
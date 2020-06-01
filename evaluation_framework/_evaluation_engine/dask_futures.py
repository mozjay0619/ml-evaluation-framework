from dask.distributed import Client, LocalCluster
from dask_yarn import YarnCluster
from evaluation_framework.utils.decorator_utils import yarn_directory_normalizer

import threading
import queue
import socket
import os
import time

def get_host_ip_address():
    """Get the host ip address of the machine where the executor is running. 

    Returns
    -------
    host_ip : String
    """
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_ip
    except:
        host_ip = '127.0.0.1'  # playing with fire...
        return host_ip


class MultiThreadTaskQueue(queue.Queue):
    
    def __init__(self, num_threads=1):
        queue.Queue.__init__(self)
        self.num_threads = num_threads
        self.start_threads()
        self.results = []
        
    def put_task(self, task, *args, **kwargs):
        self.put((task, args, kwargs))
        
    def start_threads(self):
        for i in range(self.num_threads):
            t = threading.Thread(target=self.task_in_thread)
            t.setDaemon(True)
            t.start()
            
    def task_in_thread(self):
        while True:
            task, args, kwargs = self.get()
            result = task(*args, **kwargs)
            self.results.append(result)
            self.task_done()

    def get_results(self):
        return self.results


class DualClientFuture():
    
    def __init__(self, local_client_n_workers, local_client_threads_per_worker,
                 yarn_client_n_workers, yarn_client_worker_vcores, yarn_client_worker_memory, verbose=True):
        
        host_ip = get_host_ip_address()
        
        self.local_cluster = LocalCluster(
            n_workers=local_client_n_workers,
            threads_per_worker=local_client_threads_per_worker, 
            processes=True, 
            host=host_ip)
        self.local_client = Client(address=self.local_cluster, timeout='2s') 
        
        self.yarn_cluster = YarnCluster(
            n_workers=yarn_client_n_workers, 
            worker_vcores=yarn_client_worker_vcores, 
            worker_memory=yarn_client_worker_memory,
            environment="python:///usr/bin/python3")
        self.yarn_client = Client(self.yarn_cluster)

        self.wait_container_resource_alloc()
        
        self.local_client_n_workers = local_client_n_workers
        self.yarn_client_n_workers = yarn_client_n_workers
        
        self.task_counter = -1
        self.yarn_client_n_workers = yarn_client_n_workers

        self.verbose = verbose

    def wait_container_resource_alloc(self):

        while True:
            
            waiting_containers = [yarn_container_obj for yarn_container_obj in self.yarn_cluster.workers() 
                                  if str(yarn_container_obj.state)=='WAITING']
            
            if len(waiting_containers)==0:
                break

            time.sleep(1.0)

    def submit(self, func, *args, **kwargs):

        if self.verbose==True:
            print('total n workers: {}'.format(self.local_client_n_workers + self.yarn_client_n_workers))

        self.task_counter += 1

        remainder = self.task_counter % (self.local_client_n_workers + self.yarn_client_n_workers)



        
        # if remainder <= (self.local_client_n_workers-1):

        #     if self.verbose==True:
        #         print('remainder: {}, n_local_worker: {}, running on local'.format(remainder, self.local_client_n_workers))

        #     future = self.local_client.submit(func, *args, **kwargs)
        # else:

        #     if self.verbose==True:
        #         print('remainder: {}, n_local_worker: {}, running on remote'.format(remainder, self.local_client_n_workers))

        #     func = yarn_directory_normalizer(func)
        #     future = self.yarn_client.submit(func, None, *args, **kwargs)


        if remainder <= (self.yarn_client_n_workers-1):

            if self.verbose==True:
                print('remainder: {}, n_local_worker: {}, running on remote'.format(remainder, self.local_client_n_workers))

            func = yarn_directory_normalizer(func)
            future = self.yarn_client.submit(func, None, *args, **kwargs)

        else:

            if self.verbose==True:
                print('remainder: {}, n_local_worker: {}, running on local'.format(remainder, self.local_client_n_workers))

            future = self.local_client.submit(func, *args, **kwargs)


            
        
        
        return future.result()
    
    def get_worker_ip_addresses(self):
        
        while True:
            yarn_container_objects = self.yarn_cluster.workers()
            if len(yarn_container_objects)==self.yarn_client_n_workers:
                break
            time.sleep(0.1)
            
        ip_addrs = set()
        
        for yarn_container_object in yarn_container_objects:
            ip_addrs.add(yarn_container_object.yarn_node_http_address.split('.')[0].replace('-', '.')[3:])
        
        return list(ip_addrs)
    
    def submit_per_node(self, func, *args, **kwargs):
        
        func = yarn_directory_normalizer(func)

        ip_addrs = self.get_worker_ip_addresses()

        futures = list()
        
        for ip_addr in ip_addrs:
            futures.append(self.yarn_client.submit(func, ip_addr, *args, **kwargs, workers=ip_addr))
        
        return self.yarn_client.gather(futures)
    
    def get_dashboard_link(self):
        
        print('local cluster: ', self.local_cluster.dashboard_link)
        print('yarn cluster:  ', self.yarn_cluster.dashboard_link)


class ClientFuture():
    
    def __init__(self, local_client_n_workers, local_client_threads_per_worker):
        
        host_ip = get_host_ip_address()
        self.local_cluster = LocalCluster(n_workers=local_client_n_workers,
                               threads_per_worker=local_client_threads_per_worker, 
                               processes=True, 
                               host=host_ip)
        self.local_client = Client(address=self.local_cluster, timeout='2s') 
        
    def submit(self, func, *args, **kwargs):
        
        future = self.local_client.submit(func, *args, **kwargs)
        return future.result() 
        
    def get_dashboard_link(self):
        
        print('local cluster: ', self.local_cluster.dashboard_link)


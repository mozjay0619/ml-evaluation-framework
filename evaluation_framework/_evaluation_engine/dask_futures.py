from dask.distributed import Client, LocalCluster
from dask_yarn import YarnCluster

import threading
import queue
import socket
import os

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
                 yarn_client_n_workers, yarn_client_worker_vcores, yarn_client_worker_memory):
        
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
        
        self.local_client_n_workers = local_client_n_workers
        self.yarn_client_n_workers = yarn_client_n_workers
        
        self.task_counter = 0
        self.yarn_client_n_workers = yarn_client_n_workers
        
    def submit(self, *args, **kwargs):
        
        remainder = self.task_counter % (self.local_client_n_workers + self.yarn_client_n_workers)
        
        if remainder <= (self.local_client_n_workers-1):
            future = self.local_client.submit(*args, **kwargs)
        else:
            future = self.yarn_client.submit(*args, **kwargs)
            
        self.task_counter += 1
        
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
    
    def submit_yarnworkers(self, *args, **kwargs):
        
        ip_addrs = self.get_worker_ip_addresses()
        
        futures = list()
        for ip_addr in ip_addrs:
            futures.append(self.yarn_client.submit(*args, **kwargs, workers=ip_addr))
        
        return self.yarn_client.gather(futures)
    
    def get_dashboard_links(self):
        
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
        
    def submit(self, *args, **kwargs):
        
        future = self.local_client.submit(*args, **kwargs)
        return future.result() 
        
    def get_dashboard_link(self):
        
        print('local cluster: ', self.local_cluster.dashboard_link)


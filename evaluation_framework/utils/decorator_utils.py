import functools
import time
import os

def failed_method_retry(method, max_retries=15):
    """This is a decorator for allowing certain methods to retry itself in cases of
    failures, as many as max_retries times. This decorator can work together with 
    the above two artificial traceback utility methods. 

    Example usage:

        @object_exception_catcher
        @failed_method_retry
        def run_transformer(self, df, hist_df):

        This will first allow the method to be retried, and then after max_retries times
        of retries, the exception catcher decorator will catch the final exception.
    """
    
    @functools.wraps(method)
    def failed_method_retried(*args, **kwargs):
        
        error = None
        
        for i in range(max_retries):
            
            try:
                return method(*args, **kwargs)
            
            except Exception as e:
                error = e
                print('[{}] method failed due to {}'.format(method.__name__, error))
                time.sleep(1.5)
                continue
            
        else:
            raise error
    
    return failed_method_retried


def yarn_directory_normalizer(base_method):
    
    @functools.wraps(base_method)
    def method_modifier(ip_addr, *args, **kwargs):
        """The ip_addr is for [ submit_per_node ] method in Dask Yarn dual client mode. 
        We pass in different arguments so that Dask's work assignment policy does not override
        the desired behavior of allocating the works across different host ip addresses.
        """
        try:
            
            cwd = os.getcwd()
            
            if os.path.split(cwd)[-1].find('container_')==0:
                app_root_dir = os.path.split(cwd)[0:-1][0]
                
                if os.path.split(app_root_dir)[-1].find('application_')==0:
                    os.chdir(app_root_dir)
            
            try:
                return base_method(*args, **kwargs, ip_addr=ip_addr)
            
            except:
                return base_method(*args, **kwargs)
        
        except Exception as e:
            raise type(e)(str(e) + ' from {} method'.format(base_method.__name__))
        
    return method_modifier



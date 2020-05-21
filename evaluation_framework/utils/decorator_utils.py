import functools
import time

def failed_method_retry(method, max_retries=5):
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
                time.sleep(1.0)
                continue
            
        else:
            raise error
    
    return failed_method_retried

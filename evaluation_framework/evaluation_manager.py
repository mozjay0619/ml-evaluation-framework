
from ._evaluation_manager.config_setter import ConfigSetter 
from ._evaluation_manager.method_setter import MethodSetter

# REQUIRED_FIELDS = ['estimator', 'data', 'target_name', 'feature_names', 
#               'cross_validation_scheme', 'hyperparameters',
#               'orderby', 'train_window', 'test_window', 'groupby',
#               'local_directory_path', 'S3_path',
                   
#               'str_types', 'datetime_types', 'numeric_types',
                   
#               'num_types_needed', 'missing_keys', 
#               'hdf5_filepath']

class EvaluationManager():
    
    def __init__(self):
        
        self.config_setter = ConfigSetter()
        self.method_setter = MethodSetter()
        
    def setup_evaluation(self, **kwargs):

        configs_set = self.config_setter.set_configs(**kwargs)
        methods_set = self.method_setter.set_methods(config_setter=self.config_setter, **kwargs)

        # if not self.config_setter.set_configs(**kwargs):
        #    return

        # if not self.method_setter.set_methods(config_setter=self.config_setter, **kwargs):
        #    return

        self.load_object_fields(self.config_setter)
        self.load_object_fields(self.method_setter)
        # kill objects
        
    def load_object_fields(self, source_obj):
        
        for k, v in source_obj.__dict__.items():
            # if k in REQUIRED_FIELDS:
            self.__dict__[k] = v
        
        
        
        

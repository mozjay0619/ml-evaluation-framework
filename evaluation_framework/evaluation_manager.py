from ._evaluation_manager.config_setter import ConfigSetter 
from ._evaluation_manager.method_setter import MethodSetter


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
            self.__dict__[k] = v

            
from .evaluation_manager_core.config_setter import ConfigSetter 
from .evaluation_manager_core.method_setter import MethodSetter



CONFIG_SETTER_KEYS = [
"estimator",
"data",
"target_name",
"feature_names",
"hyperparameters",
"cross_validation_scheme",
"groupby",
"orderby",
"train_window",
"min_train_window",
"test_window",
"user_configs",
"local_directory_path",
"S3_path",
"return_predictions"
]

METHOD_SETTER_KEYS = [
"preprocess_train_data",
"preprocess_test_data",
"model_fit",
"model_predict",
"store_prediction",
"evaluate_prediction"
]

FINAL_CONFIG_KEYS = [
'data',
'groupby',
'orderby'
]


class EvaluationManager():
    
    def __init__(self):
        
        self.config_setter = ConfigSetter()
        self.method_setter = MethodSetter()

        self.local_data_saved = False
        
    def setup_evaluation(self, **kwargs):

        # configs_set = self.config_setter.set_configs(**kwargs)
        # methods_set = self.method_setter.set_methods(config_setter=self.config_setter, **kwargs)

        if not self.config_setter.set_configs(local_data_saved=self.local_data_saved, **kwargs):
           return

        if not self.method_setter.set_methods(config_setter=self.config_setter, **kwargs):
           return

        self.load_object_fields(self.config_setter)
        self.load_object_fields(self.method_setter)
        # kill objects

    def update_setup(self, **kwargs):

        # must have used eval engine first, otherwise no use
        # can't update data or groupby or order by
        
        for k in FINAL_CONFIG_KEYS:
            if k in kwargs:

                raise ValueError('You cannot update the following setup arguments: [ {} ]'.format(
                            ', '.join(FINAL_CONFIG_KEYS)))

        temp_dict = dict()

        for k in CONFIG_SETTER_KEYS:
            
            temp_dict[k] = self.__dict__[k]

        for k in METHOD_SETTER_KEYS:

            temp_dict[k] = self.__dict__[k]

        for k, v in kwargs.items():
            
            temp_dict[k] = v

        

        self.setup_evaluation(**temp_dict)


        
    def load_object_fields(self, source_obj):
        
        for k, v in source_obj.__dict__.items():
            self.__dict__[k] = v













            
from ..utils.pandas_utils import is_date_str_type
from ..utils.pandas_utils import is_not_date_str_type
from ..utils.pandas_utils import is_numeric_type
from ..utils.pandas_utils import is_datetime_type
from ..utils.pandas_utils import is_float32_type
from ..utils.pandas_utils import cast_datetime2int64
from ..utils.pandas_utils import cast_numeric2float32
from ..utils.pandas_utils import encode_date_sequence
from ..utils.data_structure_utils import get_merged_list_from_dict_list_values
from ..utils.data_structure_utils import dict_is_nested
from ..utils.datetime_utils import check_date_format

import inspect
import pandas as pd
import numpy as np
import os
import warnings
import shutil
import datetime

from evaluation_framework import constants


ORDERED_CV_SCHEMES = ['date_rolling_window']
CV_OPTIONAL_ARGUMENTS = ['orderby', 'train_window', 'min_train_window', 'test_window']
OPTIONAL_ARGUMENTS = ['groupby', 'hyperparameters', 'user_configs', 'S3_path', 'user_configs', 'return_predictions']
CV_SCHEME_OPTIONS = ['date_rolling_window', 'k_fold', 'binary_classification']
INTERNAL_ARGUMENTS = ['prediction_records_dirname']
REQUIRED_ESTIMATOR_MEMBER_METHODS = ['fit', 'predict']
FIT_METHOD_PARAMETERS_PARAMETER_NAME = 'parameters'


class ConfigSetter():
    
    def __init__(self):
        
        self.dummy_region_uuid = constants.EF_DUMMY_GROUP_COLUMN_NAME

        self.str_types = []
        self.datetime_types = []
        self.numeric_types = []
        self.original_colnames = []

        self.job_uuid = str(datetime.datetime.now()).replace(" ", '-')
        
    def set_configs(self, *, local_data_saved=False, estimator=None,
                    data=None, target_name=None, feature_names=None, 
                    hyperparameters=None, cross_validation_scheme=None,
                    groupby=None, 
                    orderby=None, train_window=None, min_train_window=None, test_window=None,
                    user_configs=None, local_directory_path=None, S3_path=None, 
                    return_predictions=None, **kwargs):

        self.local_data_saved = local_data_saved
        
        self.estimator = estimator
        self.data = data
        self.target_name = target_name
        self.feature_names = feature_names
        self.hyperparameters = hyperparameters
        self.cross_validation_scheme = cross_validation_scheme
        self.groupby = groupby
        self.orderby = orderby
        self.train_window = train_window
        self.min_train_window = min_train_window
        self.test_window = test_window 
        self.user_configs = user_configs
        self.local_directory_path = local_directory_path
        self.S3_path = S3_path
        self.return_predictions = return_predictions



        print("\u2714 Checking configs requirements...    ", end="", flush=True)

        if not self.passed_arguments_requirements():
            return False
        else:
            print("Passed!")

        print("\u2714 Checking configs validity...        ", end="", flush=True)

        if not self.passed_arguments_validity():
            return False
        else:
            print("Passed!")

        self.define_helper_columns()

        

        return True

    def define_helper_columns(self):

        # create key column to join the predictions
        key_column = np.arange(len(self.data)).astype(np.float64)
        self.data[constants.EF_UUID_NAME] = key_column

        # assuming data validation was done before...
        self.numeric_types.append(constants.EF_UUID_NAME)

        if(self.return_predictions):

            self.prediction_records_dirname = 'prediction_arrays'

            # makes sense to join the predictions on self.data since that data is already
            # loaded in memory. It would be expensive to re-load it into memory, esp. if it
            # is really big.

        if(self.orderby):

            self.data[constants.EF_ORDERBY_NAME] = encode_date_sequence(self.data[self.orderby])
        
    def passed_arguments_requirements(self):
        
        self.required_args = self._required_arguments()
        self.ordered_CV_required_args = self._ordered_CV_required_arguments()
        self.optional_args = self._optional_arguments()

        if len(self.required_args)>0 or len(self.ordered_CV_required_args)>0:
            print('Failed!')
            self._tell_required_arguments()
            return False
        else:
            return True

    def _tell_required_arguments(self):

        if len(self.required_args)>0:
            print('\nRequired config argument(s):\n\n\u25BA {}'.format('  '.join(self.required_args)))
        
        if 'cross_validation_scheme' in self.required_args:
            print('\nAvailable [ cross_validation_scheme ] options:\n\n'
                  '\u25BA {}'.format('  '.join(CV_SCHEME_OPTIONS)))
            
        elif self.cross_validation_scheme in ORDERED_CV_SCHEMES:
            if len(self.ordered_CV_required_args)>0:
                print('\nRequired argument(s) for [ {} ] scheme:\n\n'
                      '\u25BA {}'.format(self.cross_validation_scheme, 
                                         '  '.join(self.ordered_CV_required_args)))

        if len(self.required_args)>0 or len(self.ordered_CV_required_args)>0:
            if len(self.optional_args)>0:
                print('\nOptional config argument(s):\n\n\u25BA {}'.format('  '.join(self.optional_args)))
                print()

    def _required_arguments(self):
        
        required_args = []
        for k, v in self.__dict__.items():
            if v is None and k not in OPTIONAL_ARGUMENTS + CV_OPTIONAL_ARGUMENTS + INTERNAL_ARGUMENTS:
                required_args.append(k)
                
        return required_args
    
    def _ordered_CV_required_arguments(self):
        
        ordered_CV_required_args = []
        
        if self.cross_validation_scheme is None:
            return ordered_CV_required_args
        
        elif self.cross_validation_scheme in ORDERED_CV_SCHEMES:
            
            for k, v in self.__dict__.items():
                
                if v is None and k in CV_OPTIONAL_ARGUMENTS:
                    
                    ordered_CV_required_args.append(k)
                
        return ordered_CV_required_args
        
    def _optional_arguments(self):
        
        args_present = []
        
        for k, v in self.__dict__.items():
            if v is not None:
                args_present.append(k)
        
        cv_optional_args = list(set(CV_OPTIONAL_ARGUMENTS) - set(self._ordered_CV_required_arguments()))
        optional_args = cv_optional_args + OPTIONAL_ARGUMENTS
        optional_args = set(optional_args) - set(args_present)   
        return optional_args

    def passed_arguments_validity(self):
        
        self._validate_user_configs()
        self._validate_local_directory_path()
        self._validate_s3_path()
        self._validate_estimator()
        self._validate_helper_columns()

        if not self.local_data_saved:

            self._validate_data()
            # self.data = self.data.astype(type_dict)
            # is creating a copy
            # we need some other way to deal with this

        self._validate_target_name()
        self._validate_feature_names()
        self._validate_hyperparameters()
        self._validate_return_predictions()
        return True

    def _validate_return_predictions(self):

        self.prediction_records_dirname = None

        if self.return_predictions is None:
            self.return_predictions = False

        else:

            if not isinstance(self.return_predictions, bool):
                raise TypeError('[ return_predictions ] must be boolean but instead got '
                                '{}'.format(type(self.return_predictions)))

            # if self.return_predictions:

            #    self.prediction_records_dirname = 'prediction_arrays'

            #    # makes sense to join the predictions on self.data since that data is already
            #    # loaded in memory. It would be expensive to re-load it into memory, esp. if it
            #    # is really big.

            #    # create key column to join the predictions
            #    key_column = np.arange(len(self.data)).astype(np.float32)
            #    self.data[constants.EF_UUID_NAME] = key_column

            #    # assuming data validation was done before...
            #    self.numeric_types.append(constants.EF_UUID_NAME)

    def _validate_user_configs(self):

        if self.user_configs is None:

            self.user_configs = dict()

    def _validate_local_directory_path(self):
        """
        [ local_directory_path ] validation involves checking its existence. If it
        exists, pass. If it does not exist, create one. As a convention, it is 
        encouraged to name it as os.path.join(os.getcwd(), "evaluation_tasks")
        [ local_directory_path ] is the root path for all subsequent evaluation tasks, 
        including the current one. Each evaluation task is given uuid, defined by the 
        run datetime (milliseconds). 
        [ evaluation_task_dirpath ] is the directory for the current evaluation task. 
        Its name is to be prefixed by "evaluation_task__{}".format(job_uuid).
        [ job_uuid ] is the current datetime in milliseconds.  

        Defined fields:
        ---------------

        initial_dirpath
        job_uuid
        evaluation_task_dirpath
        memmap_root_dirname
        memmap_root_dirpath

        Directory structures:
        ---------------------

        local_directory_path

        |__ evaluation_task_dirname (relative) / evaluation_task_dirpath (absolute)

            |__ memmap_root_dirname (relative) / memmap_root_dirpath (absolute)

                |__ memmap_map
                |__ groupA__groupA'__arrayA

            |__ 

        |__ evaluation_task_dirname (relative) / evaluation_task_dirpath (absolute)

        Path creation sequence:
        -----------------------

        Nothing is created at [ EvaluationManager ]
        [ evaluation_task_dirname ] directory created at [ run_evaluation ] method of [ EvaluationEngine ]

        """ 
        if os.path.exists(self.local_directory_path):
            pass
        else:
            os.makedirs(self.local_directory_path)
            
        self.initial_dirpath = os.getcwd()
        

        self.evaluation_task_dirname = "evaluation_task__{}".format(self.job_uuid)
        self.evaluation_task_dirpath = os.path.join(self.local_directory_path, self.evaluation_task_dirname)
        
        self.memmap_root_dirname = 'memmap_root_dir'
        self.memmap_root_dirpath = os.path.join(self.evaluation_task_dirpath, self.memmap_root_dirname)

    def _validate_s3_path(self):

        self.memmap_root_S3_object_name = self.memmap_root_dirname + '__' + self.job_uuid
        
    def _validate_estimator(self):
        
        member_methods = inspect.getmembers(self.estimator, predicate=inspect.ismethod)
        member_methods = [elem[0] for elem in member_methods]
        
        if not set(REQUIRED_ESTIMATOR_MEMBER_METHODS) < set(member_methods):
            missing_member_methods = set(REQUIRED_ESTIMATOR_MEMBER_METHODS) - set(member_methods)
            print('Failed!')
            raise ValueError('[ estimator ] object is missing \"{}\" '
                             'method(s).'.format(', '.join(missing_member_methods)))
            
        fit_method_parameters = inspect.signature(self.estimator.fit).parameters.keys()
        
        if FIT_METHOD_PARAMETERS_PARAMETER_NAME in fit_method_parameters:
            
            if self.hyperparameters is None:
                print('Failed!')
                raise ValueError('[ estimator ] object requires [ hyperparameters ] argument.')

        else:

            if self.hyperparameters is not None:
                warnings.warn("[ hyperparameters ] is not being used by [ estimator ].")

    def _validate_helper_columns(self):

        if constants.EF_UUID_NAME in self.data:
            self.data.drop(labels=constants.EF_UUID_NAME, axis=1, inplace=True)
            
        if constants.EF_ORDERBY_NAME in self.data:
            self.data.drop(labels=constants.EF_ORDERBY_NAME, axis=1, inplace=True)
            
        if constants.EF_PREDICTION_NAME in self.data:
            self.data.drop(labels=constants.EF_PREDICTION_NAME, axis=1, inplace=True)
        
    def _validate_target_name(self):
        
        if self.target_name not in self.data.columns:
            print('Failed!')
            raise ValueError('[ target_name ] "{}" is not in the dataframe.'.format(self.target_name))
            
        if not is_float32_type(self.data[self.target_name]):
            print('Failed!')
            raise TypeError('[ target_name ] "{}" is not of type np.float32.'.format(self.target_name))
            
    def _validate_feature_names(self):
            
        if isinstance(self.feature_names, dict):
            self._check_groupby_presence('feature_names')
            self._check_groupby_contents('feature_names')
        
        # make it into dict if it is not
        else:
            if self.groupby:
                group_keys = self.data[self.groupby].unique()
                feature_names = self.feature_names
                self.feature_names = {group_key: feature_names for group_key in group_keys}
            else:
                self.data[self.dummy_region_uuid] = self.dummy_region_uuid
                self.feature_names = {self.dummy_region_uuid: self.feature_names}
            
        feature_names_list = get_merged_list_from_dict_list_values(self.feature_names)
        
        for feature_name in feature_names_list:
            if not is_float32_type(self.data[feature_name]):
                print('Failed!')
                raise TypeError('[ feature_names ] "{}" is not of type np.float32,'.format(feature_name))
            
        for k, v in self.feature_names.items():
            
            # check the feature names in dataframe
            if not set(v) < set(self.data.columns):
                
                unrecognized_feature_names = set(v) - set(self.data.columns)
                
                if k == self.dummy_region_uuid:
                    print('Failed!')
                    raise ValueError("[ feature_names ] \"{}\" are missing from the input "
                                     "dataframe.".format(', '.join(unrecognized_feature_names)))
                else:
                    print('Failed!')
                    raise ValueError("[ feature_names ] \"{}\" are missing from the input "
                                     "dataframe at group \"{}\".".format(', '.join(unrecognized_feature_names), k))
      
    def _validate_data(self):
        
        if not isinstance(self.data, pd.DataFrame):
            print('Failed!')
            raise TypeError('[ data ] must be of type <pandas.DataFrame>, '
                            'instead got {}.'.format(type(self.data)))
        
        if len(self.data)==0:
            print('Failed!')
            raise ValueError('[ data ] is empty dataframe.')

        self.data = self.data.reset_index(drop=True)
            
        self.str_types = []
        self.date_str_types = []
        self.datetime_types = []
        self.numeric_types = []
        self.unsupported_types = []
        
        types = self.data.dtypes

        for k, v in types.items():

            if v=='object' and is_not_date_str_type(self.data[k]):
                self.str_types.append(k)
            elif v=='object' and is_date_str_type(self.data[k]):
                self.date_str_types.append(k)
            elif v=='object':
                # raise warning!
                self.str_types.append(k)
            elif is_datetime_type(self.data[k]):
                self.datetime_types.append(k)
            elif is_numeric_type(self.data[k]):
                self.numeric_types.append(k)
            else:
                self.unsupported_types.append(k)
                
        if len(self.unsupported_types)>0:
            print('Failed!')
            raise TypeError('has unsupported types')   ## FIX
            
        if self.orderby:
            if self.orderby not in self.data.columns:
                print('Failed!')
                raise ValueError('The [ orderby ] "{}" column name'
                                 'does not exist in [ data ] dataframe.'.format(self.orderby))

            if self.cross_validation_scheme=='date_rolling_window':
                if self.orderby not in self.datetime_types and self.orderby not in self.date_str_types:
                    print('Failed!')
                    raise TypeError('[ orderby ] argument column in [ data ] dataframe must have either '
                                    'datetime dtype or "YYYY-MM-DD" format if '
                                    '[ cross_validation_scheme ] is specified as "date_rolling_window".')

        if self.groupby:
            if self.groupby not in self.data.columns:
                print('Failed!')
                raise ValueError('The [ groupby ] "{}" column name'
                                 'does not exist in [ data ] dataframe.'.format(self.groupby))

        # convert all date str types to datetime types    
        while len(self.date_str_types)>0:
            # raise warning!
            k = self.date_str_types.pop()
            self.data[k] = pd.to_datetime(self.data[k])
            self.datetime_types.append(k)

        # convert all numeric to float32
        # for k in self.numeric_types:
        #     # raise warning!
        #     self.data[k] = cast_numeric2float32(self.data[k])

        type_dict = {elem: np.float32 for elem in self.numeric_types}
        self.data = self.data.astype(type_dict)

        self.original_colnames = self.data.columns.to_list()
    
    def _validate_hyperparameters(self):
        
        if self.hyperparameters is None:
            return
        
        if isinstance(self.hyperparameters, list):
            self.hyperparameters = {self.dummy_region_uuid: self.hyperparameters} # check fit

            return
        
        if not isinstance(self.hyperparameters, dict):
            print('Failed!')
            raise TypeError('[ hyperparameters ] argument must be either list or dict, '
                            'instead got {}'.format(type(self.hyperparameters)))
            
        if dict_is_nested(self.hyperparameters):
            
            self._check_groupby_presence('hyperparameters')
            self._check_groupby_contents('hyperparameters')
            self._check_orderby_presence('hyperparameters')
                                    
            # check that the dates can be converted to datetime type
            self._check_orderby_contents('hyperparameters')
                
            # check the date range
            # there should be overlaps
            
        else:  # single level dict 
            
            if not self.cross_validation_scheme in ORDERED_CV_SCHEMES:
                # if not, there is no way this is dated keys
                
                self._check_groupby_presence('hyperparameters')
                self._check_groupby_contents('hyperparameters')
                    
            else:
                # if it is ordered, we may have dated key or grouped key
                
                if any([check_date_format(elem) 
                        for elem in list(self.hyperparameters.keys())]):
                    
                    self._check_orderby_presence('hyperparameters')
                    self._check_orderby_contents('hyperparameters')
                        
                    # we need to include the dummy group
                    self.hyperparameters = {self.dummy_region_uuid: self.hyperparameters}
                    
                elif len(set(self.hyperparameters.keys())
                         .intersection(set(self.data[self.groupby].unique()))) > 0:
                    
                    self._check_groupby_presence('hyperparameters')
                    self._check_groupby_contents('hyperparameters')
                    
                    # we need to include the dummy order
                    
                else:
                    print('Failed!')
                    raise ValueError('asdf')
                    
                
    def _check_groupby_presence(self, parameter_name):
        
        if self.groupby is None:
            print('Failed!')
            raise ValueError('If you are specifying per group [ {} ] you '
                             'must supply [ groupby ] argument.'.format(parameter_name))
            
    def _check_groupby_contents(self, parameter_name):
        """Used only when parameter_name parameter is a dictionary type, and against
        the [ data ] dataframe [ groupby ] column contents."""
        
        d = self.__dict__[parameter_name]
        
        if set(d.keys())!=set(self.data[self.groupby].unique()):
            print('Failed!')
            raise ValueError('The [ groupby ] column values of [ data ] dataframe and '
                             'the group keys of [ {} ] dict do not match.'.format(parameter_name))
            
    def _check_orderby_presence(self, parameter_name):
        
        if self.orderby is None:
            print('Failed!')
            raise ValueError('If you are specifying per date [ {} ] you '
                             'must supply [ orderby ] argument.'.format(parameter_name))
            
    def _check_orderby_contents(self, parameter_name):
        
        d = self.__dict__[parameter_name]
        
        if dict_is_nested(d):

            dates = set()
            for k, v in d.items():
                for k_, v_ in v.items():
                    dates.add(k_)
            dates = list(dates)
            
        else:
            
            dates = list(d.keys())
            
        if not all([check_date_format(elem) for elem in dates]):
            print('Failed!')
            raise ValueError('The date keys of [ {} ] failed to be parsed. '
                             'They must be of the form YYYY-MM-DD.'.format(parameter_name)) 

            
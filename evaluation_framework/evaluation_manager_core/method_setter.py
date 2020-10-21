from ..task_graph.default_methods import default_preprocess_train_data
from ..task_graph.default_methods import default_preprocess_test_data
from ..task_graph.default_methods import default_model_fit
from ..task_graph.default_methods import default_model_predict
from ..task_graph.default_methods import default_evaluate_prediction

import re
import copy


REQUIRED_METHOD_ARGUMENTS = ['preprocess_train_data', 'preprocess_test_data', 'evaluate_prediction']
EVALUATION_OPTIONS = ['mse', 'mae', 'rmse']


class MethodSetter():
	
	def __init__(self):
		
		self.num_types_needed = None
		self.missing_keys = dict()
		self.missing_keys['datetime_types'] = []
		self.missing_keys['str_types'] = []
		
	def set_methods(self, config_setter=None, 
					preprocess_train_data="", preprocess_test_data="",
					model_fit=None, model_predict=None,
					store_prediction=None, evaluate_prediction="", **kwargs):

		if config_setter.groupby is None:
			config_setter.groupby = config_setter.dummy_region_uuid
		
		self.config_setter = config_setter
		self.preprocess_train_data = preprocess_train_data
		self.preprocess_test_data = preprocess_test_data
		
		self.model_fit = model_fit
		self.model_predict = model_predict
		self.store_prediction = store_prediction
		self.evaluate_prediction = evaluate_prediction
		
		print("\u2714 Checking methods requirements...    ", end="", flush=True)
		if not self.passed_arguments_requirements():
			return False
		else:
			print('Passed!')

		print("\u2714 Checking methods validity...        ", end="", flush=True)
		if not self.passed_arguments_validity():
			return False
		else:
			print('Passed!')
		
		return True
	
	def passed_arguments_requirements(self):

		# if self.config_setter is None:
		# 	print('You must')
		# 	return False

		requirements = copy.copy(REQUIRED_METHOD_ARGUMENTS)
		
		if self.preprocess_train_data!="":
			requirements.remove('preprocess_train_data')
		if self.preprocess_test_data!="":
			requirements.remove('preprocess_test_data')
		if self.evaluate_prediction!="":
			requirements.remove('evaluate_prediction')

		if len(requirements)>0:
			print('Failed!')
			print('\nRequired method argument(s):\n\n\u25BA {}'.format('  '.join(requirements)))

			if 'evaluate_prediction' in requirements:
				print('\nAvailable predefined [ evaluate_prediction ] options:\n\n'
                  '\u25BA {}'.format('  '.join(EVALUATION_OPTIONS)))

			print('\nIf you pass in None, default methods will be used instead.')

			return False
		else:
			return True

	def passed_arguments_validity(self):
		
		# need to check they passed in arguments correctly
		# need to check that they do not require configs if nothing is passed in
		
		
		self._validate_preprocess_train_data_method()

		self._validate_preprocess_test_data()
		self._validate_model_fit()
		self._validate_model_predict()
		# self._validate_store_prediction()
		self._validate_evaluate_prediction()

		self.num_types_needed = self._num_types_needed()

		return True
		
		
	def _validate_preprocess_train_data_method(self):
		
		if self.preprocess_train_data is None:
			
			self.preprocess_train_data = default_preprocess_train_data
			return
	
	def _validate_preprocess_test_data(self):
		
		if self.preprocess_test_data is None:
			
			self.preprocess_test_data = default_preprocess_test_data
			return
	
	def _validate_model_fit(self):
		
		if self.model_fit is None:
			
			self.model_fit = default_model_fit
			return
	
	def _validate_model_predict(self):
		
		if self.model_predict is None:
			
			self.model_predict = default_model_predict
			return
	
	def _validate_store_prediction(self):
		if self.store_prediction is None:
			
			self.store_prediction = default_store_prediction
			return
	
	def _validate_evaluate_prediction(self):
		if self.evaluate_prediction is None:
			
			self.evaluate_prediction = default_evaluate_prediction
			return
		
	def _num_types_needed(self):

		self.sample_train_pdf, self.sample_test_pdf = self._get_sample_pdf(self.config_setter)

		included_colnames = copy.copy(self.config_setter.numeric_types)
		train_missing_keys, re = self.key_error_catcher(
			self.preprocess_train_data, 
			self.sample_train_pdf[included_colnames], 
			self.config_setter.user_configs)
		test_missing_keys, re = self.key_error_catcher(
			self.preprocess_test_data, 
			self.sample_test_pdf[included_colnames], 
			re,
			self.config_setter.user_configs)
		missing_keys = train_missing_keys + test_missing_keys
		
		if len(missing_keys)==0:
			return 1

		self.missing_keys['datetime_types'] = copy.copy(missing_keys)
		
		if len(missing_keys)>0:
			
			included_colnames += self.config_setter.datetime_types
			train_missing_keys, re = self.key_error_catcher(
				self.preprocess_train_data, 
				self.sample_train_pdf[included_colnames], 
				self.config_setter.user_configs)
			test_missing_keys, re = self.key_error_catcher(
				self.preprocess_test_data, 
				self.sample_test_pdf[included_colnames], 
				re, 
				self.config_setter.user_configs)
			missing_keys = train_missing_keys + test_missing_keys
			
		if len(missing_keys)==0:
			return 2

		self.missing_keys['datetime_types'] = list(set(self.missing_keys['datetime_types']) - set(missing_keys))
		self.missing_keys['str_types'] = copy.copy(missing_keys)
			
		if len(missing_keys)>0:
			
			included_colnames += self.config_setter.str_types
			train_missing_keys, re = self.key_error_catcher(
				self.preprocess_train_data, 
				self.sample_train_pdf[included_colnames], 
				self.config_setter.user_configs)
			test_missing_keys, re = self.key_error_catcher(
				self.preprocess_test_data, 
				self.sample_test_pdf[included_colnames], 
				re, 
				self.config_setter.user_configs)
			missing_keys = train_missing_keys + test_missing_keys
			
		if len(missing_keys)==0:
			return 3
		
		else:
			raise ValueError('not included! {}'.format(missing_keys))

	def _get_sample_pdf(self, config_setter):

		# do the groupby size ordering here! and get the smallest one!
		# by the str! that we can use it later

		if config_setter.groupby:
			
			sample_group = config_setter.data[config_setter.groupby].iloc[0]
			
			sample_pdf = config_setter.data[config_setter.data[config_setter.groupby]==sample_group]
			
			n = min(len(sample_pdf), 1000)
			train_n = int(n * 0.75)
			
			sample_train_pdf = sample_pdf.iloc[0:train_n]
			sample_test_pdf = sample_pdf.iloc[train_n:]
			
		else:

			n = min(len(config_setter.data), 1000)
			train_n = int(n * 0.75)

			sample_pdf = config_setter.data.iloc[0:n]
			sample_train_pdf = sample_pdf.iloc[0:train_n]
			sample_test_pdf = sample_pdf.iloc[train_n:]

		return sample_train_pdf, sample_test_pdf

	def key_error_catcher(self, f, *args, **kwargs):
	
		try:
			re = f(*args, **kwargs)
			return [], re

		except KeyError as e:

			key = e.args[0]

			if key in self.config_setter.original_colnames:
				return [key], None

			key = eval(re.search('(\[\'.*\'\])', key, re.IGNORECASE).group(1))

			if set(key) < set(self.config_setter.original_colnames):
				return key, None
			
			return key, None
			

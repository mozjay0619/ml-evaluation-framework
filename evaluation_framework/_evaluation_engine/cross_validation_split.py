from abc import ABCMeta, abstractmethod
import numpy as np

from sklearn.model_selection._split import *
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

__all__ = ['BaseCrossValidator',
           'KFold',
           'GroupKFold',
           'LeaveOneGroupOut',
           'LeaveOneOut',
           'LeavePGroupsOut',
           'LeavePOut',
           'RepeatedStratifiedKFold',
           'RepeatedKFold',
           'ShuffleSplit',
           'GroupShuffleSplit',
           'StratifiedKFold',
           'StratifiedShuffleSplit',
           'PredefinedSplit',
           'train_test_split',
           'check_cv',
           'BaseRollingWindowSplit',
           'DateRollingWindowSplit']


def get_cv_splitter(cross_validation_scheme, *args, **kwargs):

    if cross_validation_scheme=='date_rolling_window':
        return DateRollingWindowSplit(*args, **kwargs)
    elif cross_validation_scheme=='k_fold':
        return KFold(*args, **kwargs)


class BaseRollingWindowSplit(metaclass=ABCMeta):
    
    def __init__(self, orderby=None, random_state=None):
        self.orderby = orderby
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        for train, test, date_range in self._iter_indices(X, y, groups):
            yield train, test, date_range
    
    @abstractmethod
    def _iter_indices(self, X, y=None, groups=None):
        """Generate (train, test) indices"""
    
    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the total number of splits"""
    
    def __repr__(self):
        return _build_repr(self)
        

class DateRollingWindowSplit(BaseRollingWindowSplit):
    
    def __init__(self, num_train_days, num_test_days, min_num_train_days, orderby,
                 random_state=None):
        super().__init__(random_state=random_state, orderby=orderby)
        self.num_train_days = num_train_days
        self.num_test_days = num_test_days
        self.min_num_train_days = min_num_train_days
    
    def _iter_indices(self, X, y, groups):
        
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        head_date_idx = self.orderby.min()
        last_date_idx = self.orderby.max()

        while(head_date_idx + self.num_train_days <= last_date_idx):

            train = indices[(head_date_idx <= self.orderby) & 
                            (self.orderby < head_date_idx + self.num_train_days)]

            test = indices[(head_date_idx + self.num_train_days <= self.orderby) & 
                           (self.orderby < head_date_idx + self.num_train_days + self.num_test_days)]

            current_head_date_idx = head_date_idx

            head_date_idx += self.num_test_days

            num_unique_days = len(np.unique(self.orderby[train]))

            if(num_unique_days<self.min_num_train_days or len(test)==0):

                continue

            yield(train, test, 
                np.arange(current_head_date_idx + self.num_train_days, 
                    min(current_head_date_idx + self.num_train_days + self.num_test_days, last_date_idx+1)))
            
    def get_n_splits(self, X=None, y=None, groups=None):
        
        split_cnt = 0
        
        for i in self._iter_indices(X=self.orderby, y=None, groups=None):
            split_cnt+=1
            
        return split_cnt

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)


def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values()
                       if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", FutureWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, 'cvargs'):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category == FutureWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))


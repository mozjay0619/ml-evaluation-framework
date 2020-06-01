
__version__ = "0.0.b31"



from ._evaluation_manager.config_setter import ConfigSetter
from ._evaluation_manager.method_setter import MethodSetter

from .evaluation_manager import EvaluationManager

from ._evaluation_engine.data_loader import load_local_data
from ._evaluation_engine.data_loader import upload_local_data
from ._evaluation_engine.data_loader import download_local_data
from ._evaluation_engine.data_loader import upload_remote_data
from ._evaluation_engine.data_loader import download_remote_data

from ._evaluation_engine.dask_futures import MultiThreadTaskQueue
from ._evaluation_engine.dask_futures import ClientFuture
from ._evaluation_engine.dask_futures import DualClientFuture
from ._evaluation_engine.cross_validation_split import DateRollingWindowSplit
from ._evaluation_engine.cross_validation_split import get_cv_splitter

from .task_graph import TaskGraph

from evaluation_framework.utils.objectIO_utils import save_obj
from evaluation_framework.utils.objectIO_utils import load_obj

from evaluation_framework.utils.memmap_utils import write_memmap
from evaluation_framework.utils.memmap_utils import read_memmap

from .evaluation_engine import EvaluationEngine

__all__ = [
	"EvaluationManager",
	"EvaluationEngine",
	]

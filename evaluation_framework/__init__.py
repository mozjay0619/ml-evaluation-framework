
__version__ = "0.0.b6"



from ._evaluation_manager.config_setter import ConfigSetter
from ._evaluation_manager.method_setter import MethodSetter

from .evaluation_manager import EvaluationManager

from ._evaluation_engine.data_loader import DataLoader
from ._evaluation_engine.dask_futures import MultiThreadTaskQueue
from ._evaluation_engine.dask_futures import ClientFuture
from ._evaluation_engine.dask_futures import DualClientFuture
from ._evaluation_engine.split import DateRollingWindowSplit
from ._evaluation_engine.split import get_cv_splitter

from ._evaluation_engine.task_graph import TaskGraph

from ._evaluation_engine.memmap_layer import read_memmap
from ._evaluation_engine.memmap_layer import load_obj

from .evaluation_engine import EvaluationEngine

__all__ = [
	"ConfigSetter",
	"DataLoader",
	]

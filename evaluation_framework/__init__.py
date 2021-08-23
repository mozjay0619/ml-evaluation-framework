
__version__ = "0.0.b71"



from .evaluation_manager_core.config_setter import ConfigSetter
from .evaluation_manager_core.method_setter import MethodSetter

from .evaluation_manager import EvaluationManager

# from .evaluation_engine_core.data_loader import load_local_data

from .evaluation_engine_core.data_transferer import upload_local_data
from .evaluation_engine_core.data_transferer import download_local_data
from .evaluation_engine_core.data_transferer import upload_remote_data
from .evaluation_engine_core.data_transferer import download_remote_data

# from .evaluation_engine_core.parallel.dask_client_base import MultiThreadTaskQueue
# from .evaluation_engine_core.parallel.dask_client_base import ClientFuture
# from .evaluation_engine_core.parallel.dask_client_base import DualClientFuture

from .task_graph.cross_validation_split import DateRollingWindowSplit
from .task_graph.cross_validation_split import get_cv_splitter

from .task_graph.task_graph import TaskGraph

from evaluation_framework.utils.objectIO_utils import save_obj
from evaluation_framework.utils.objectIO_utils import load_obj

from evaluation_framework.utils.memmap_utils import write_memmap
from evaluation_framework.utils.memmap_utils import read_memmap

from .evaluation_engine import EvaluationEngine

__all__ = [
	"EvaluationManager",
	"EvaluationEngine",
	]

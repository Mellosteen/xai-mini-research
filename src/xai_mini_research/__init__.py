"""Mini project for XAI and Clever Hans behavior in time-series regression."""
from .config import get_default_config_path
from .data import generate_time_data
from .preprocessing import preprocess
from .metrics import regression_metrics, regression_metrics_all_splits

__version__ = "0.1.0"
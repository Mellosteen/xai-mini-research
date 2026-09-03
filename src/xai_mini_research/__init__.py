"""Mini project for XAI and Clever Hans behavior in time-series regression."""
from .config import get_default_config_path
from .data import generate_time_data
from .preprocessing import preprocess
from .metrics import regression_metrics, regression_metrics_all_splits
from .results import save_results
from .explain import explain_mlp_lrp, summarize_lrp_relevance, summarize_mlp_lrp
from .interventions import intervene_scaled_shortcut

__version__ = "0.1.0"

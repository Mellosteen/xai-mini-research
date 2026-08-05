from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

def get_default_config_path() -> Path:
    """
    Returns the default config path.
    """
    return DEFAULT_CONFIG_PATH
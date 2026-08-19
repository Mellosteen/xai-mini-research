import json
from pathlib import Path
from datetime import datetime

def save_results(results : dict, filename : str | None = None):
    """
    Creates a results directory if none exist and saves results from compare_models.py
    """
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"model_comparison_{timestamp}.json"

    output_path = results_dir / filename

    with output_path.open("w") as file:
        json.dump(results, file, indent=4)

    return output_path
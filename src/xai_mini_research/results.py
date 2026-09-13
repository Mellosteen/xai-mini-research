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

def save_experiment_details(project_root, artifact_dir, run_id, activation, activation_label):
    """
    Save a copy of the source code and record which setup was used for this run.

    Args:
        project_root (Path): Root directory of the repository.
        artifact_dir (Path): Directory for this run's source copy and saved models.
        run_id (String): Timestamp with the activation suffix.
        activation (String): Activation argument, such as 'lgsigmoid'.
        activation_label (String): Display name, such as 'LogSigmoid'.

    Returns:
        experiment_details (Dict): Code revision, local changes, source hashes, package
            versions, and command. Saved under 'provenance' in the result JSON.
    """
    import hashlib
    import importlib.metadata
    import subprocess
    import zipfile
    import torch

    artifact_dir.mkdir(parents=True, exist_ok=False)
    code_files = sorted((project_root / "src").rglob("*.py"))
    code_files.append(project_root / "requirements.txt")
    code_files.append(project_root / "configs/default.yaml")

    # Keep the actual files used, since a Git commit does not include local changes.
    source_path = artifact_dir / "source.zip"
    source_hashes = {}
    with zipfile.ZipFile(source_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code_files:
            relative_path = path.relative_to(project_root).as_posix()
            file_contents = path.read_bytes()
            source_hashes[relative_path] = hashlib.sha256(file_contents).hexdigest()
            archive.write(path, relative_path)

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project_root, text=True,
    ).strip()

    package_versions = {}
    for package in ("torch", "numpy", "scikit-learn", "zennit", "matplotlib"):
        package_versions[package] = importlib.metadata.version(package)

    experiment_details = {
        "run_id": run_id,
        "activation": activation_label,
        "activation_slug": activation,
        "git_commit": git_commit,
        "working_tree_dirty": bool(git_status),
        "git_status_at_start": git_status.splitlines(),
        "source_sha256": source_hashes,
        "source_archive": source_path.relative_to(project_root).as_posix(),
        "versions": package_versions,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "command": f"PYTHONPATH=src python -m xai_mini_research.experiments.compare_models --activation {activation}",
    }
    return experiment_details


def save_mlp_training(model, processed_data, training_metrics, project_root, artifact_dir, model_name):
    """
    Save the trained MLP weights and the scaler needed to prepare its inputs again.

    Args:
        model (MLPRegressor): Trained baseline or shortcut model.
        processed_data (Dict): Matching data, including the fitted scaler in metadata.
        training_metrics (MLPTrainingMetrics): Training and validation losses per epoch.
        project_root (Path): Root directory, used to save a relative checkpoint path.
        artifact_dir (Path): Existing output directory for the current run.
        model_name (String): 'baseline' or 'shortcut', used in output filenames.

    Returns:
        training_summary (Dict): Completed epochs, loss lists, and checkpoint path.
            A checkpoint contains the weights and architecture settings needed to reload the model.
    """
    import numpy as np
    import torch

    activation = model.activation_name
    checkpoint_path = artifact_dir / f"{model_name}_{activation}.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "activation": activation,
        "input_dim": processed_data["train"]["X_scaled"].shape[1],
        "hidden_dimensions": [model.network[0].out_features, model.network[2].out_features],
    }
    torch.save(checkpoint, checkpoint_path)

    # These arrays contain the fitted StandardScaler parameters, not new estimates.
    scaler = processed_data["metadata"]["scaler"]
    scaler_path = artifact_dir / f"{model_name}_scaler_{activation}.npz"
    np.savez(
        scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_samples_seen=scaler.n_samples_seen_,
        feature_names=np.asarray(processed_data["metadata"]["feature_names"]),
    )

    training_summary = {
        "completed_epochs": len(training_metrics.train_losses),
        "train_losses": training_metrics.train_losses,
        "val_losses": training_metrics.val_losses,
        "checkpoint": checkpoint_path.relative_to(project_root).as_posix(),
    }
    return training_summary

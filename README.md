# xai-mini-research

An introductory controlled study of shortcut reliance in synthetic temporal regression. Linear regression, RBF kernel ridge regression, and a small MLP are evaluated on clean inputs and inputs augmented with a deliberately leaked polynomial feature. Zennit LRP explanations and feature interventions help investigate the MLP's dependence on that shortcut.

The saved experiments show that a useful-looking shortcut can fail severely during extrapolation, and that changing the shortcut substantially changes predictions. The study is a small research-learning project; its findings are limited to the tested synthetic conditions.

## Start here

- [Methods, architecture, and results summary](reports/Final_Report.md): experiment design, numerical findings, limitations, and selected evidence.
- [Research logs](reports/research_logs.md): dated implementation notes and evolving interpretations.
- [Saved results](results/): metrics, relevance summaries, and figure paths for individual runs.
- **Formal report:** the author will write this separately in Overleaf using LaTeX. Links to its source and PDF will be added when available.

The core experiment and a comparison of ReLU, LogSigmoid, and GELU have saved results. Direct hidden-layer measurements explain the ReLU shortcut model's flat prediction region in the selected run. The [summary](reports/Final_Report.md#checking-the-relu-flatline) distinguishes this regional inactivity from claims about permanently inactive neurons or general activation benefits.

## Setup

Clone the repository and create an environment:

```bash
git clone https://github.com/Mellosteen/xai-mini-research.git
cd xai-mini-research
python -m venv venv
```

Activate it on macOS/Linux with `source venv/bin/activate`, or on Windows with `venv\Scripts\activate`, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

For the existing local development environment, use `conda activate mini-holiday` instead of creating a new environment.

## Run the current experiment

From the repository root with the environment active:

```bash
# macOS / Linux
PYTHONPATH=src python -m xai_mini_research.experiments.compare_models --activation lgsigmoid
```

In Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m xai_mini_research.experiments.compare_models --activation lgsigmoid
```

Choose `relu`, `lgsigmoid` (LogSigmoid), or `gelu`; omitting the option retains GELU as the default. Run the command once for each desired activation. Each call trains clean and shortcut models, selects kRR settings, evaluates MLP relevance and interventions, and measures hidden activations.

Results and every figure include the activation suffix, for example `model_comparison_<timestamp>_lgsigmoid.json` and `lrp_line_shortcut_<timestamp>_lgsigmoid.png`. JSON files are saved in `results/`, figures in `reports/`, and checkpoints, scaler parameters, raw measurements, and a source snapshot in `results/artifacts_<timestamp>_<activation>/`. Figures also label the activation inside the image.

The effective settings are in [compare_models.py](src/xai_mini_research/experiments/compare_models.py), with MLP defaults in [mlp.py](src/xai_mini_research/models/mlp.py). [configs/default.yaml](configs/default.yaml) is a setup placeholder and does not control this comparison. The command runs the selected activation using current settings; it does not run all three activations automatically. Saved September 13 results record the actual code and settings; older unlabelled files have incomplete provenance. See the [summary's saved-result notes](reports/Final_Report.md#saved-results-and-verification) before comparing outputs.

## Run tests

With the environment active, from the repository root:

```bash
python -m pytest -q
```

The suite checks data generation, splitting, preprocessing, models, metrics, explanations, and interventions. Passing implementation tests does not establish statistical robustness or explanation faithfulness.

## Repository layout

```text
src/xai_mini_research/
  data.py, preprocessing.py       Synthetic data, shortcut, and scaling
  models/                        Linear regression, kRR, and MLP
  explain.py, interventions.py    Relevance and shortcut interventions
  diagnostics.py                 Hidden activation measurements and plots
  experiments/compare_models.py  Experiment settings, evaluation, and plots
  metrics.py, results.py          Metrics and JSON output
reports/                         Technical summary, research logs, and figures
results/                         Timestamped experiment records
configs/                         Setup configuration placeholder
tests/                           Implementation tests
```

## Author

Austin Samuel Qiu
